import uvicorn

from autogpt.api.settings import settings


def main() -> None:
    uvicorn.run(
        "autogpt.api.application:get_app",
        workers=settings.workers_count,
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        factory=settings.factory,
    )


if __name__ == "__main__":
    main()
