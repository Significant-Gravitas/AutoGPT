from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Optional


class LeadCreate(BaseModel):
    """
    Schema para crear un nuevo lead.
    """
    nombre: str = Field(..., min_length=2, max_length=255, description="Nombre completo del lead")
    email: EmailStr = Field(..., description="Email del lead")
    empresa: Optional[str] = Field(None, max_length=255, description="Nombre de la empresa")
    sector: Optional[str] = Field(None, max_length=100, description="Sector empresarial")
    mensaje: Optional[str] = Field(None, description="Mensaje o consulta del lead")

    class Config:
        json_schema_extra = {
            "example": {
                "nombre": "Juan Pérez",
                "email": "juan.perez@empresa.com",
                "empresa": "Empresa XYZ",
                "sector": "Retail",
                "mensaje": "Me interesa conocer sus servicios de automatización"
            }
        }


class LeadResponse(BaseModel):
    """
    Schema para la respuesta de un lead.
    """
    id: int
    nombre: str
    email: str
    empresa: Optional[str]
    sector: Optional[str]
    mensaje: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True
