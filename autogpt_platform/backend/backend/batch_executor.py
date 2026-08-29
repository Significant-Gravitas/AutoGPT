from backend.app import run_processes
from backend.executor.batch_executor import BatchExecutor


def main():
    """
    Run the BatchExecutor poll loop as a standalone service.

    This is the entry point the infra repo deploys (the loop otherwise
    only runs inside the all-in-one ``poetry run app`` process, which
    k8s never uses — services are deployed individually).
    """
    run_processes(
        BatchExecutor(),
    )


if __name__ == "__main__":
    main()
