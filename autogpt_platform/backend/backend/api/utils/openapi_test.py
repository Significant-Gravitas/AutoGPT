from fastapi import FastAPI, File, UploadFile

from backend.api.utils.openapi import sort_openapi


def test_file_upload_fields_keep_binary_format():
    app = FastAPI()

    @app.post("/upload")
    async def upload(file: UploadFile = File(...), files: list[UploadFile] = File(...)):
        return {}

    sort_openapi(app)
    schemas = app.openapi()["components"]["schemas"]
    props = next(s for s in schemas.values() if "file" in s.get("properties", {}))[
        "properties"
    ]

    assert props["file"] == {"type": "string", "format": "binary", "title": "File"}
    assert props["files"]["items"] == {"type": "string", "format": "binary"}
