import os
import ssl
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field, TypeAdapter, field_validator

from backend.util.link_checkout.models import StrictModel


class BrokerRoute(StrictModel):
    url: str
    ca: str
    client_cert: str
    client_key: str
    secret_file: str

    @field_validator("url")
    @classmethod
    def fixed_origin(cls, value: str) -> str:
        value = value.rstrip("/")
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path
        ):
            raise ValueError("Checkout broker requires a fixed HTTPS origin")
        return value

    def tls_context(self) -> ssl.SSLContext:
        context = ssl.create_default_context(cafile=self.ca)
        context.load_cert_chain(self.client_cert, self.client_key)
        return context

    def credential(self) -> str:
        secret = Path(self.secret_file).read_text().strip()
        if len(secret) < 32:
            raise ValueError("Checkout controller credential is too short")
        return secret


class TenantRoute(BrokerRoute):
    user_id: str = Field(min_length=1, max_length=128)


def configured() -> bool:
    """Whether any user's browser runs in a remote broker."""
    return bool(
        os.environ.get("CHECKOUT_BROKER_ROUTES_FILE")
        or os.environ.get("CHECKOUT_BROKER_URL")
    )


def routed(user_id: str) -> bool:
    """Whether this user has a broker. Everyone else browses as usual."""
    try:
        route_for(user_id)
    except (OSError, ValueError, KeyError):
        return False
    return True


def route_for(user_id: str) -> BrokerRoute:
    if registry := os.environ.get("CHECKOUT_BROKER_ROUTES_FILE"):
        with Path(registry).open("rb") as source:
            data = source.read(1_000_001)
        if len(data) > 1_000_000:
            raise ValueError("Checkout routing configuration too large")
        routes = TypeAdapter(list[TenantRoute]).validate_json(data)
        if len({route.user_id for route in routes}) != len(routes):
            raise ValueError("Duplicate checkout tenant routes")
        matches = [route for route in routes if route.user_id == user_id]
        if len(matches) != 1:
            raise ValueError("Private checkout is not provisioned for this user")
        return matches[0]
    # One broker, for the one user it was provisioned for; it refuses anyone
    # else, so nobody else is sent there.
    if not user_id or user_id != os.environ.get("CHECKOUT_BROKER_USER_ID"):
        raise ValueError("Private checkout is not provisioned for this user")
    return BrokerRoute(
        url=os.environ["CHECKOUT_BROKER_URL"],
        ca=os.environ["CHECKOUT_BROKER_CA"],
        client_cert=os.environ["CHECKOUT_BROKER_CLIENT_CERT"],
        client_key=os.environ["CHECKOUT_BROKER_CLIENT_KEY"],
        secret_file=os.environ["CHECKOUT_BROKER_SECRET_FILE"],
    )
