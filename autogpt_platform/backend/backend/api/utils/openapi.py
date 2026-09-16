from fastapi import FastAPI


def sort_openapi(app: FastAPI) -> None:
    """
    Patch a FastAPI instance's `openapi()` method to sort the endpoints,
    schemas, and responses.
    """
    wrapped_openapi = app.openapi

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = wrapped_openapi()

        # Sort endpoints
        openapi_schema["paths"] = dict(sorted(openapi_schema["paths"].items()))

        # Sort endpoints -> methods
        for p in openapi_schema["paths"].keys():
            openapi_schema["paths"][p] = dict(
                sorted(openapi_schema["paths"][p].items())
            )

            # Sort endpoints -> methods -> responses
            for m in openapi_schema["paths"][p].keys():
                openapi_schema["paths"][p][m]["responses"] = dict(
                    sorted(openapi_schema["paths"][p][m]["responses"].items())
                )

        # Sort schemas and responses as well
        for k in openapi_schema["components"].keys():
            openapi_schema["components"][k] = dict(
                sorted(openapi_schema["components"][k].items())
            )

        _restore_binary_format(openapi_schema)

        app.openapi_schema = openapi_schema
        return openapi_schema

    app.openapi = custom_openapi


def _restore_binary_format(node: object) -> None:
    # fastapi>=0.129.1 renders file fields as contentMediaType; orval and other
    # client generators only map `format: binary` to a Blob/File type.
    if isinstance(node, list):
        for item in node:
            _restore_binary_format(item)
        return
    if not isinstance(node, dict):
        return
    if (
        node.get("type") == "string"
        and node.get("contentMediaType") == "application/octet-stream"
    ):
        entries = list(node.items())
        node.clear()
        node.update(
            ("format", "binary") if key == "contentMediaType" else (key, value)
            for key, value in entries
        )
    for value in node.values():
        _restore_binary_format(value)
