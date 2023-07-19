from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from jose import jwt, JWTError

# TODO : Code using USE_DICTIONARY , users_db , auth to be moved to User
from autogpt.core.user.user import User , USE_DICTIONARY , users_db
from firebase_admin import auth

class UserCreate(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    uid: str
    email: str

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/user/login")


app = FastAPI()

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
        if not user_in_db or not User.password_context.verify(user.password, user_in_db.password_hash):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
    else:
        try:
            user_in_db = auth.get_user_by_email(user.email)
        except:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    access_token = User.create_access_token(user_in_db.email)
    response = JSONResponse(content={"access_token": access_token})
    response.set_cookie(key="session", value=access_token, httponly=True, secure=True)
    return response

@app.get("/protected_route")
def protected_route(token: str = Depends(oauth2_scheme)):
    user = User.get_user_from_access_token(token)
    return {"user": user}