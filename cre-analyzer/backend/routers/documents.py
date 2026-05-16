from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional
from services.document_parser import (
    parse_offering_memorandum,
    parse_t12,
    parse_rent_roll,
    get_demo_data,
)

router = APIRouter()


@router.post("/parse")
async def parse_document(
    file: UploadFile = File(...),
    doc_type: str = Form(...),  # "om", "t12", "rent_roll"
):
    contents = await file.read()
    filename = file.filename or "upload"

    try:
        if doc_type == "om":
            result = parse_offering_memorandum(contents, filename)
        elif doc_type == "t12":
            result = parse_t12(contents, filename)
        elif doc_type == "rent_roll":
            result = parse_rent_roll(contents, filename)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown doc_type: {doc_type}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")

    return {"doc_type": doc_type, "filename": filename, "data": result}


@router.get("/demo")
def get_demo():
    """Return pre-populated demo data without requiring document upload."""
    return get_demo_data()
