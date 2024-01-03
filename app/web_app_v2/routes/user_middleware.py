from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)


class UserIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request.state.user_id = "a1621e69-970a-4340-86e7-778d82e2137b"

        LOG.info("Entering : UserIDMiddleware")

        # Get the endpoint's name
        endpoint = request.scope.get("endpoint")
        endpoint_name = endpoint.__name__ if endpoint else "Unknown"
        LOG.info("Endpoint : " + endpoint_name)
        LOG.info("Path : " + request.scope.get("path"))
        LOG.info(f"Request {request.method} {request.url}")

        # try:
        response = await call_next(request)
        # except Exception as e:
        #     # Log the exception for more information
        #     LOG.error(f"Exception encountered: {str(e)}")

        #     # Log the request data again for context
        #     LOG.error(f"Request that caused exception: {request.method} {request.url}\n- Body: {request_body.decode()}")

        #     # Re-raise the exception to ensure the error propagates and is handled appropriately
        #     raise e

        LOG.info(
            f"Response {request.method} {request.url} - Status: {response.status_code}"
        )
        LOG.info("Exiting : UserIDMiddleware")
        return response
