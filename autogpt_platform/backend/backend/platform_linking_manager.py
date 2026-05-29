from backend.app import run_processes
from backend.platform_linking.manager import PlatformLinkingManager


def main():
    """
    Run the AutoGPT-server Platform Linking Manager service.
    """
    run_processes(
        PlatformLinkingManager(),
    )


if __name__ == "__main__":
    main()
