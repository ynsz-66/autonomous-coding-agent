import json
import logging
from pathlib import Path
from typing import Any
import time
import fire
import os
from datetime import datetime
from dotenv import load_dotenv
from openai import APIStatusError, RateLimitError
from agent.docker import pull_image, start_container, cleanup, exec_command
from rich.live import Live
from rich.panel import Panel
from agent.prompts import format_swebench_prompt
from mcp_config.client_swebench import list_tools, make_builtins
from sandbox import SandboxConfig
from validators.swe_models import (
    SWEBenchTaskInput,
    StepMetrics,
    SolutionOutput,
)
from agent.llm import OpenAICompatibleLLM
from utils import (
    CONSOLE, FinishError,
    load_llm_config, rotate_on_error,
    extract_code, format_observation
)
from ui import (
    LogPanelHandler, build_layout, render_thinking, render_sandbox,
    render_metrics, render_summary, wait_for_enter,
)


load_dotenv()


def init_docker(docker_image: str, logger) -> Any:
    """Pull the task image and start the container for SWE-bench.

    Args:
        docker_image: Fully qualified SWE-bench task image.

    Returns:
        The running Docker container handle.
    """
    CONSOLE.print(f"[grey37]Pulling {docker_image}...[/]")
    logger.info(f"Pulling the docker image ({docker_image})...")
    pull_image(docker_image)
    logger.info(f"Docker image {docker_image} ready")

    code_dir = str(Path(__file__).resolve().parent)  # src/
    CONSOLE.print(f"[grey37]Starting container ({code_dir} -> /agent)[/]")
    logger.info("Starting container...")
    container = start_container(docker_image, code_dir)
    logger.info(
        f"Container {container.short_id} started "
        f"({code_dir} mounted on /agent)")

    CONSOLE.print("[grey37]Installing dependencies inside the container...[/]")
    logger.info("Installing dependencies inside the container...")
    exit_code, output = exec_command(
        container, "python -m pip install -r /agent/agent/requirements.txt")
    if exit_code != 0:
        msg = (
            "pip install -r /agent/agent/requirements.txt failed in the "
            f"container (exit code {exit_code}):\n{output[-2000:]}")
        logger.critical(msg)
        CONSOLE.print(
            Panel(
                f"[red]pip install failed:\n{output}[/]",
                border_style="red"))
        raise RuntimeError(msg)

    logger.info("Container ready, MCP tool dependencies installed")
    CONSOLE.print(
        Panel(
            "[green]Container ready.[/]",
            border_style="green",
            expand=False))

    return container


def run(task_file: str,
        output: str,
        max_iterations: int = 30,
        max_execution_time_seconds: int = 900,
        max_input_tokens: int = 300000,
        max_output_tokens: int = 10000,
        pause: bool = False,
        ) -> None:
    """Run the SWE-bench agent loop on a single task.

    Args:
        task_file: Path to the JSON task input.
        output: Path where the resulting ``solution.json`` is written.
        max_iterations: Hard cap on agent turns.
        max_execution_time_seconds: Wall-clock budget for the whole
            run (time spent paused is excluded).
        max_input_tokens: Accumulated input-token budget.
        max_output_tokens: Accumulated output-token budget.
        pause: Wait for Enter after each sandbox result (demo/debug).
    """
    container = None
    logger = None
    file_handler = None
    panel_handler = LogPanelHandler()
    steps: list[StepMetrics] = []
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_requests: int = 0
    started = time.time()
    paused: float = 0.0
    result_panel: Panel | None = None
    output_path: Path = Path(output)
    try:
        base_url, model_name, api_keys = load_llm_config()
        os.system('clear')
        CONSOLE.print(
            Panel(
                "[bold magenta]AGENT SWEBENCH DEMO[/]",
                border_style="purple",
                expand=False))
        CONSOLE.print("")

        task_path: Path = Path(task_file)

        with task_path.open("r") as f:
            task = SWEBenchTaskInput(**json.load(f))

        os.system("mkdir -p logs")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        logger = logging.getLogger(f"swebench_{task.instance_id}_{timestamp}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        file_handler = logging.FileHandler(
            f"logs/swebench_{task.instance_id}_{timestamp}.log", mode="w")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)
        logger.addHandler(panel_handler)
        container = init_docker(task.docker_image, logger)
        tools = list_tools(container.id)

        sandbox = SandboxConfig(
            extra_builtins=make_builtins(
                tools, container.id))
        schemas = [{"name": t.name,
                    "description": t.description or "",
                    "inputSchema": t.input_schema} for t in tools.tools]
        system = format_swebench_prompt(
            problem_statement=task.problem_statement,
            repo=task.repo,
            hints_text=task.hints_text,
            tool_schemas=schemas,
            authorized_imports=sandbox.authorized_imports,
            max_iterations=max_iterations
        )

        iteration: int = 0
        key_idx: int = 0
        llm = OpenAICompatibleLLM(api_keys[0], base_url, model_name)
        provider_failures: int = 0
        history: list[dict[str, Any]] = [{"role": "user", "content": "Begin."}]
        retries: int = 0
        layout = build_layout()
        started = time.time()

        def refresh_panels(waiting: bool = False) -> None:
            layout["logs"].update(panel_handler.render())
            layout["footer"].update(render_metrics(
                iteration, max_iterations,
                total_input_tokens, max_input_tokens,
                total_output_tokens, max_output_tokens,
                time.time() - started - paused, max_execution_time_seconds,
                total_requests, retries, paused, llm.model, waiting))

        with Live(layout, console=CONSOLE, refresh_per_second=20, screen=True):
            while iteration < max_iterations:
                elapsed = time.time() - started - paused
                if elapsed > max_execution_time_seconds:
                    msg = (f"Time budget exceeded: {elapsed:.0f}s elapsed > "
                           f"{max_execution_time_seconds}s allowed "
                           f"(max_execution_time_seconds)")
                    logger.critical(msg)
                    raise TimeoutError(msg)
                request_time = time.time()
                logger.info(
                    f"── Iteration {iteration + 1}/{max_iterations} ──")
                logger.info(
                    f"Sending request to {llm.base_url} "
                    f"(model: {llm.model}, API key #{key_idx + 1})...")
                refresh_panels()
                try:
                    response = llm.chat_stream(
                        system, history, stop=["<end_code>"])
                    full_response = ""
                    for text in response:
                        full_response += text
                        layout["thinking"].update(
                            render_thinking(full_response))
                        refresh_panels()
                        time.sleep(0.02)
                except (RateLimitError, APIStatusError) as e:
                    llm, key_idx, provider_failures = rotate_on_error(
                        e, llm, api_keys, key_idx, provider_failures,
                        logger.warning)
                    continue
                provider_failures = 0
                end_request = time.time()
                total_requests += 1
                request_time_ms = (end_request - request_time) * 1000
                input_tokens = response.usage["prompt_tokens"]
                output_tokens = response.usage["completion_tokens"]
                logger.info(
                    f"LLM replied in {request_time_ms:.0f} ms "
                    f"(input: {input_tokens} tokens, "
                    f"output: {output_tokens} tokens)")

                if len(full_response) == 0:
                    retries += 1
                    logger.warning(
                        "LLM returned an empty response, asking again "
                        f"(retry #{retries})")
                    continue
                code = extract_code(full_response)
                if not code.strip():
                    retries += 1
                    logger.warning(
                        "LLM reply has no python code block, asking it to "
                        f"resend one (retry #{retries})")
                    history.append(
                        {"role": "assistant", "content": full_response})
                    history.append({"role": "user", "content": (
                        "No valid Python code block found in your "
                        "reply. Every turn MUST contain exactly one "
                        "```python ... ``` block. Try again."
                    )})
                    continue
                sandbox_output = sandbox.run(code)
                layout["sandbox"].update(render_sandbox(code, sandbox_output))
                if sandbox_output.get("error"):
                    logger.warning(
                        "Sandbox error sent back to the LLM: "
                        f"{sandbox_output['error']}")

                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                if total_input_tokens > max_input_tokens:
                    msg = (f"Input token budget exceeded: "
                           f"{total_input_tokens} > {max_input_tokens} "
                           f"(max_input_tokens)")
                    logger.critical(msg)
                    raise RuntimeError(msg)
                if total_output_tokens > max_output_tokens:
                    msg = (f"Output token budget exceeded: "
                           f"{total_output_tokens} > {max_output_tokens} "
                           f"(max_output_tokens)")
                    logger.critical(msg)
                    raise RuntimeError(msg)
                steps.append(StepMetrics(
                    step=iteration + 1,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    request_time_ms=request_time_ms,
                    api_url=llm.base_url,
                    model_name=llm.model,
                    llm_output=full_response,
                    sandbox_input=code,
                    sandbox_output=json.dumps(sandbox_output),
                    retries=retries
                ))
                retries = 0

                if pause:
                    refresh_panels(waiting=True)
                    waited = wait_for_enter()
                    paused += waited
                    logger.info(
                        f"Paused by user for {waited:.1f}s "
                        "(not counted in the time budget)")
                    refresh_panels()

                if "final_answer" in sandbox_output:
                    solution = (
                        sandbox_output['stdout']
                        if sandbox_output['stdout'] != ""
                        else sandbox_output['final_answer']
                    )
                    with open(output_path, 'w') as f:
                        f.write(SolutionOutput(
                            task_id=str(task.instance_id),
                            benchmark="swebench",
                            success=True,
                            solution=solution,
                            system_prompt=system,
                            iterations=iteration + 1,
                            total_requests=total_requests,
                            total_input_tokens=total_input_tokens,
                            total_output_tokens=total_output_tokens,
                            total_time_seconds=time.time() - started - paused,
                            steps=steps
                        ).model_dump_json(indent=4))
                    logger.info(
                        f"final_answer() called, solution written to "
                        f"{output_path}")
                    raise FinishError()

                history.append({"role": "assistant", "content": full_response})
                if (
                    sandbox_output.get("stdout", "") == ""
                    and sandbox_output.get("error", "") == ""
                ):
                    observation = (
                        "Your code produced no stdout. Wrap the values you "
                        "need to inspect in print() and try again."
                    )
                else:
                    observation = format_observation(sandbox_output)
                history.append({"role": "user", "content": observation})
                iteration += 1

        if iteration >= max_iterations:
            result_panel = Panel(
                f"[bold red]Max iterations reached without final_answer() "
                f"({iteration}/{max_iterations})[/]",
                border_style="red")
            if logger:
                logger.critical(
                    f"Max iterations reached ({iteration}/{max_iterations}) "
                    "without final_answer(), no solution written")

    except FinishError:
        result_panel = Panel(
            f"[bold green]Results saved -> {output_path}[/]",
            border_style="green")
        if logger:
            logger.info("Agent finished successfully")

    except Exception as e:
        result_panel = Panel(
            "[bold red]"
            f"({e.__class__.__name__})"
            f"\n{e}[/]",
            border_style="red")
        if logger:
            logger.critical(f"Run aborted by {e.__class__.__name__}: {e}")

    except (KeyboardInterrupt, SystemExit):
        result_panel = Panel(
            "[bold red]SystemExit or KeyboardInterrupt "
            "Caught, exit program..[/]",
            border_style="red",
            expand=False)
        if logger:
            logger.critical(
                "Run interrupted (Ctrl+C or SystemExit), no solution written")

    finally:
        if steps:
            CONSOLE.print(render_summary(steps, {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "time": time.time() - started - paused,
                "paused": paused,
                "requests": total_requests,
            }))
        if result_panel:
            CONSOLE.print(result_panel)
        if logger:
            logger.removeHandler(panel_handler)
        if file_handler:
            file_handler.close()
            if logger:
                logger.removeHandler(file_handler)
        if container:
            CONSOLE.print()
            CONSOLE.print("[grey37]Cleaning the container...[/]")
            CONSOLE.print()
            cleanup(container)


def main() -> None:
    """Entry point: expose :func:`run` as a Fire CLI."""
    fire.Fire(run)


if __name__ == "__main__":
    main()
