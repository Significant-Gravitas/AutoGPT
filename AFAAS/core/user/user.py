from datetime import datetime, timedelta

from fastapi import HTTPException, status
from firebase_admin import auth, credentials, initialize_app
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, SecretStr

USE_DICTIONARY = False  # Change to True to use in-db dictionary

# Initialize Firebase
if not USE_DICTIONARY:
    cred = credentials.Certificate("path/to/serviceAccountKey.json")
    initialize_app(cred)

# TODO : Set in parameter
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 4320
JWT_RENEWAL_PERIOD_MINUTES = 1440

password_context = (
    CryptContext(schemes=["bcrypt"], deprecated="auto") if USE_DICTIONARY else None
)


class UserCreate(BaseModel):
    email: EmailStr
    password: SecretStr


class UserLogin(BaseModel):
    email: EmailStr
    password: SecretStr


class UserResponse(BaseModel):
    uid: str
    email: str


class User(BaseModel):
    email: str
    password_hash: str if USE_DICTIONARY else None

    @staticmethod
    def create_access_token(email: str) -> str:
        expires = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode = {"email": email, "exp": expires}
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    @staticmethod
    def decode_access_token(token: str) -> str:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            email = payload.get("email")
            return email
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def get_user_from_access_token(token: str):
        email = User.decode_access_token(token)
        if not email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )

        # Check if token needs to be rotated
        decoded_token = User.decode_access_token(token)
        exp_timestamp = decoded_token.get("exp")
        current_timestamp = datetime.utcnow()
        time_to_expiry = exp_timestamp - current_timestamp

        if (
            time_to_expiry.total_seconds() <= JWT_RENEWAL_PERIOD_MINUTES * 60
        ):  # If less than 24 hours remaining until expiry
            new_token = User.rotate_token(token)
            # Return the new token to be used in the client application
            return new_token

        user = users_db.get(email) if USE_DICTIONARY else auth.get_user_by_email(email)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
            )
        return user

    @staticmethod
    def rotate_token(token: str) -> str:
        decoded_token = User.decode_access_token(token)
        email = decoded_token.get("email")
        if not email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )
        new_token = User.create_access_token(email)
        return new_token


users_db = {
    "user1@example.com": (
        User(
            email="user1@example.com", password_hash=password_context.hash("password1")
        )
        if USE_DICTIONARY
        else None
    ),
    "user2@example.com": (
        User(
            email="user2@example.com", password_hash=password_context.hash("password2")
        )
        if USE_DICTIONARY
        else None
    ),
}
