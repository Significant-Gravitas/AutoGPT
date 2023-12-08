from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from firebase_admin import auth
from pydantic import BaseModel

# TODO : Code using USE_DICTIONARY , users_db , auth to be moved to User
from AFAAS.app.core.user.user import (USE_DICTIONARY, User,
                                                     UserCreate, UserLogin,
                                                     UserResponse, users_db)


class OAuth2Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/user/login/oauth2")

app = FastAPI()
# bypass_routes = ["/user/login", "/user/register"]
# app.add_middleware(JWTAuthenticationMiddleware, bypass_routes=bypass_routes)


@app.post("/user/register")
def register_user(user: UserCreate):
    try:
        if USE_DICTIONARY:
            hashed_password = User.password_context.hash(user.password)
            new_user = User(email=user.email, password_hash=hashed_password)
            users_db[user.email] = new_user
        else:
            new_user = auth.create_user(email=user.email, password=user.password)
        return UserResponse(uid=new_user.uid, email=new_user.email)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/user/login")
def login_user(user: UserLogin):
    if USE_DICTIONARY:
        user_in_db = users_db.get(user.email)
        if not user_in_db or not User.password_context.verify(
            user.password.get_secret_value(), user_in_db.password_hash
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
            )
    else:
        try:
            user_in_db = auth.get_user_by_email(user.email)
        except:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
            )

    access_token = User.create_access_token(user_in_db.email)
    refresh_token = User.create_refresh_token(user_in_db.email)
    response = JSONResponse(
        content={
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
        }
    )
    response.set_cookie(
        key="session", value=access_token, httponly=True, secure=True, samesite="Lax"
    )
    return response


@app.post("/user/login/oauth2")
def login_user_oauth2(token: str = Depends(oauth2_scheme)):
    email = User.decode_token(token)
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )

    access_token = User.create_access_token(email)
    refresh_token = User.create_refresh_token(email)
    response = OAuth2Token(
        access_token=access_token, refresh_token=refresh_token, token_type="bearer"
    )
    return response


@app.post("/refresh_token")
def refresh_token(refresh_token: str):
    email = User.decode_token(refresh_token)
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )

    access_token = User.create_access_token(email)
    response = JSONResponse(content={"access_token": access_token})
    response.set_cookie(
        key="session", value=access_token, httponly=True, secure=True, samesite="Lax"
    )
    return response


@app.get("/protected_route")
def protected_route(token: str = Depends(oauth2_scheme)):
    user = User.get_user_from_access_token(token)
    return {"user": user}
