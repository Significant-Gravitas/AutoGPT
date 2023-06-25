"""
This is a wrapper for autogpt that helps with bootstrapping, especially for the docker installation option.

Some of the checks done here are also done in the new autogpt cli (autogpt/cli.py) but we are using a separate wrapper to minimize the size of the 
installation package. We don't want to include the whole of AutoGPT since it's already included in the docker image.
"""

import click
from autogpt.config.container_config import ContainerConfig
 
@click.group(invoke_without_command=True)
@click.option("--image", help="Docker image to use when building or pulling (defaults to significantgrativas/auto-gpt).")
@click.option("--repo", help="GitHub 'repo-user/repo-name' to download config files from if they don't exist locally (defaults to 'Significant-Grativas/Auto-GPT')")
@click.option("--branch", help="GitHub branch to download config files from if they don't exist locally. (defaults to 'stable')")
@click.option("--rebuild-image", is_flag=True, help="Rebuild docker image. Useful if you've change the requirements.txt file.")
@click.option("--pull-image", is_flag=True, help="Pull new image. If the image has changed, but you don't want to delete local config.")
@click.option("--reinstall", is_flag=True, help="Delete local config and  and start fresh")
@click.argument("args", nargs=-1)
def wrapper(
    image_name: str,
    repo: str,
    branch: str,
    rebuild_image: bool,
    pull_image: bool,
    reinstall: bool,
    args: list[str],
) -> None:
    """
    Wrapper for autogpt that does scaffolding and bootstrapping, especially for the docker installation option.
    """
    continuous = "--continuous" in args or "-c" in args
    container_config = ContainerConfig(
        interactive=(not continuous), 
        image_name=image_name, 
        repo=repo, 
        branch=branch, 
        rebuild_image=rebuild_image, 
        pull_image=pull_image, 
        reinstall=reinstall, 
        allow_virtualenv=False, 
        args=args
    )
    container_config.run()

if __name__ == "__main__":
    wrapper()