"""The emails about a sign-in provider ("Continue with ..."), neither with a
link: the code that confirms an address the provider doesn't host, and the
notice that the provider can now sign in to the account. Both name the
provider, so their copy is built for each send."""

import re
from typing import Literal

from backend.notifications.auth_content import AuthEmailContent

AuthSignInEmailType = Literal["sign_in_code", "sign_in_method_added"]
# Better Auth's id for each provider, with the name the emails use.
SignInProvider = Literal["google", "microsoft", "apple", "github", "discord"]
SIGN_IN_PROVIDER_NAMES: dict[SignInProvider, str] = {
    "google": "Google",
    "microsoft": "Microsoft",
    "apple": "Apple",
    "github": "GitHub",
    "discord": "Discord",
}
# Must match the codes the frontend issues (lib/auth/sign-in-code.ts).
SIGN_IN_CODE_PATTERN = r"^[0-9]{6}$"


def sign_in_email_content(
    email_type: AuthSignInEmailType, code: str | None, provider: SignInProvider | None
) -> AuthEmailContent:
    if provider is None:
        raise ValueError(f"The {email_type} email needs a provider")
    name = SIGN_IN_PROVIDER_NAMES[provider]
    if email_type == "sign_in_method_added":
        return _method_added_content(name)
    if not (code and re.fullmatch(SIGN_IN_CODE_PATTERN, code)):
        raise ValueError("The sign_in_code email needs a six-digit code")
    return _code_content(name)


def _code_content(provider: str) -> AuthEmailContent:
    """Sent before a provider joins the existing account that uses this
    address, when the provider doesn't run the mailbox. Whoever started the
    sign-in types the code into the page that asked, so it isn't a link: the
    mailbox's owner can't approve someone else's sign-in by clicking."""
    return AuthEmailContent(
        subject="Your AutoGPT sign-in code",
        eyebrow="ACCOUNT · SIGN-IN CODE",
        headline="Confirm it's you",
        intro=(
            f"Someone is signing in to AutoGPT with {provider} using this "
            "email address."
        ),
        preheader="Enter this code on the page that asked for it to finish signing in.",
        reason=f"This code was requested by a {provider} sign-in on AutoGPT.",
        # Must match the frontend's code lifetime (sign-in-code.ts).
        expiry="10 minutes",
        note=(
            "If this wasn't you, ignore this email: nobody can sign in with "
            f"{provider} without this code. Don't share it with anyone."
        ),
        close_with_inbox=True,
    )


def _method_added_content(provider: str) -> AuthEmailContent:
    return AuthEmailContent(
        subject=f"{provider} sign-in was added to your AutoGPT account",
        eyebrow="ACCOUNT · SIGN-IN METHODS",
        headline=f"You can now sign in with {provider}",
        intro=f"{provider} sign-in was just added to your AutoGPT account.",
        preheader="If this was you, there's nothing to do.",
        reason="We email you whenever a new way to sign in is added to your account.",
        note=(
            "If it wasn't you, reply to this email and we'll remove it and "
            "secure your account."
        ),
        close_with_inbox=True,
    )
