from typing import Any

import docker


def pull_image(image: str) -> None:
    """Pull a Docker image from the registry.

    Args:
        image: Fully qualified image name (with tag).
    """
    client = docker.from_env()
    client.images.pull(image, platform="linux/amd64")


def start_container(image: str, code_dir: str) -> Any:
    """Start a detached container with the local code directory mounted.

    Args:
        image: Docker image to launch.
        code_dir: Host directory to bind-mount as `/agent` inside.

    Returns:
        The created Docker container handle.
    """
    client = docker.from_env()
    volumes = {code_dir: {"bind": "/agent", "mode": "rw"}}
    return client.containers.run(
        image,
        command="sleep infinity",
        detach=True,
        tty=True,
        volumes=volumes,
        working_dir="/agent",
    )


def cleanup(container: Any) -> None:
    """Stop and remove the given Docker container.

    Args:
        container: Container handle returned by `start_container`.
    """
    container.stop()
    container.remove()


def exec_command(container: Any, command: str) -> tuple[int, str]:
    """Run a shell command inside the container and capture its output.

    Args:
        container: Container handle returned by `start_container`.
        command: Shell command to execute inside the container.

    Returns:
        A pair `(exit_code, decoded_output)`.
    """
    result = container.exec_run(command)
    return result.exit_code, result.output.decode()
