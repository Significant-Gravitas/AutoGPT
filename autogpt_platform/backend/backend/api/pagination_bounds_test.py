"""Every paginated route bounds `page` and `page_size` in its schema.

A page number past what a query's 64-bit offset can carry reached Postgres and
answered 500. A new paginated route fails
`test_every_page_parameter_has_an_upper_bound` until it bounds both, with
`MAX_PAGE` and `MAX_PAGE_SIZE` from `backend.util.models` or a tighter limit.
"""

from collections import defaultdict
from collections.abc import Iterator

from fastapi import FastAPI
from starlette.routing import Mount

from backend.api.rest_api import app

ALLOWED_UNBOUNDED = {
    # Capping these would change the public v1 contract.
    "GET /external-api/v1/store/agents page_size",
    "GET /external-api/v1/store/creators page_size",
}


def test_every_page_parameter_has_an_upper_bound():
    unbounded = {
        f"{operation} {name}"
        for operation, name, maximum in _page_parameters()
        if maximum is None
    }

    assert unbounded <= ALLOWED_UNBOUNDED, (
        "Give these an upper bound (le=MAX_PAGE / le=MAX_PAGE_SIZE from "
        f"backend.util.models): {sorted(unbounded - ALLOWED_UNBOUNDED)}"
    )
    assert ALLOWED_UNBOUNDED <= unbounded, (
        "These are bounded now, so drop them from ALLOWED_UNBOUNDED: "
        f"{sorted(ALLOWED_UNBOUNDED - unbounded)}"
    )


def test_every_bounded_offset_fits_in_64_bits():
    bounds: dict[str, dict[str, int]] = defaultdict(dict)
    for operation, name, maximum in _page_parameters():
        if maximum is not None:
            bounds[operation][name] = maximum

    overflowing = [
        operation
        for operation, b in bounds.items()
        if {"page", "page_size"} <= b.keys()
        and (b["page"] - 1) * b["page_size"] >= 2**63
    ]
    assert not overflowing, f"(page - 1) * page_size overflows: {overflowing}"


def _page_parameters() -> Iterator[tuple[str, str, int | None]]:
    """("METHOD /path", name, maximum) for every served page/page_size query."""
    for prefix, served in _apps():
        for path, operations in served.openapi()["paths"].items():
            for method, operation in operations.items():
                for param in operation.get("parameters", []):
                    if param["in"] == "query" and param["name"] in (
                        "page",
                        "page_size",
                    ):
                        yield (
                            f"{method.upper()} {prefix}{path}",
                            param["name"],
                            param["schema"].get("maximum"),
                        )


def _apps(served: FastAPI = app, prefix: str = "") -> Iterator[tuple[str, FastAPI]]:
    yield prefix, served
    for route in served.routes:
        if isinstance(route, Mount) and isinstance(route.app, FastAPI):
            yield from _apps(route.app, prefix + route.path)
