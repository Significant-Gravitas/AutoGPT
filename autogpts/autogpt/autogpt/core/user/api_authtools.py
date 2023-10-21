from fastapi.responses import JSONResponse
from fastapi import FastAPI, BackgroundTasks
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from autogpts.autogpt.autogpt.core.user.middleware_jwt import JWTAuthenticationMiddleware
from autogpts.autogpt.autogpt.core.user.user import UserLogin
from firebase_admin import auth


app = FastAPI()
# bypass_routes = ["/user/login", "/user/register"]
# app.add_middleware(JWTAuthenticationMiddleware, bypass_routes=bypass_routes)


conf = ConnectionConfig(
    MAIL_USERNAME="YourSMTPUsername",
    MAIL_PASSWORD="YourSMTPPassword",
    MAIL_FROM="YourEmail",
    MAIL_PORT=587,
    MAIL_SERVER="YourSMTPServer",
    MAIL_TLS=True,
    MAIL_SSL=False,
    USE_CREDENTIALS=True,
)


@app.post("/user/password_reset")
async def password_reset(user: UserLogin, background_tasks: BackgroundTasks):
    action_code_settings = {
        "url": "http://localhost:8000/user/password_reset_confirm",
        "handle_code_in_app": True,
    }
    link = auth.generate_password_reset_link(user.email, action_code_settings)
    email = MessageSchema(subject="Password Reset", recipients=[user.email], body=link)
    fm = FastMail(conf)
    background_tasks.add_task(fm.send_message, email)
    return JSONResponse(
        status_code=200,
        content={"message": "Password reset link has been sent to the provided email."},
    )


@app.post("/user/email_verify")
async def email_verify(user: UserLogin, background_tasks: BackgroundTasks):
    action_code_settings = {
        "url": "http://localhost:8000/user/email_verify_confirm",
        "handle_code_in_app": True,
    }
    link = auth.generate_email_verification_link(user.email, action_code_settings)
    email = MessageSchema(
        subject="Email Verification", recipients=[user.email], body=link
    )
    fm = FastMail(conf)
    background_tasks.add_task(fm.send_message, email)
    return JSONResponse(
        status_code=200,
        content={
            "message": "Email verification link has been sent to the provided email."
        },
    )
