from fastapi import HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from starlette.middleware.base import (BaseHTTPMiddleware,
                                       RequestResponseEndpoint)

from AFAAS.app.core.user.user import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/user/login")

from fastapi import HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from starlette.middleware.base import (BaseHTTPMiddleware,
                                       RequestResponseEndpoint)


class JWTAuthenticationMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, bypass_routes=[]):
        super().__init__(app)
        self.bypass_routes = bypass_routes

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        if str(request.url.path) not in self.bypass_routes:
            token = request.cookies.get("session")
            if not token:
                if "Authorization" in request.headers:
                    scheme, param = request.headers["Authorization"].split()
                    if scheme.lower() == "bearer":
                        token = param
            if token:
                try:
                    payload = User.decode_access_token(token)
                    request.state.user = payload
                except JWTError:
                    raise HTTPException(
                        status_code=403, detail="Could not validate credentials"
                    )
        response = await call_next(request)
        return response
