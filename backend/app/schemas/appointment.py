from pydantic import BaseModel, Field, EmailStr
from datetime import datetime
from typing import Optional


class AppointmentCreate(BaseModel):
    """
    Schema para crear una nueva cita de diagnóstico.
    """
    # Información del lead (puede ser existente o nuevo)
    nombre: str = Field(..., min_length=2, max_length=255)
    email: EmailStr
    empresa: Optional[str] = Field(None, max_length=255)
    sector: Optional[str] = Field(None, max_length=100)

    # Información de la cita
    fecha_preferida: datetime = Field(..., description="Fecha y hora preferida para el diagnóstico")
    servicio_interes: Optional[str] = Field(None, max_length=255, description="Servicio de interés")
    mensaje: Optional[str] = Field(None, description="Mensaje adicional")

    class Config:
        json_schema_extra = {
            "example": {
                "nombre": "María García",
                "email": "maria.garcia@empresa.com",
                "empresa": "Retail Corp",
                "sector": "Retail",
                "fecha_preferida": "2025-11-15T10:00:00",
                "servicio_interes": "Automatización de procesos",
                "mensaje": "Me gustaría conocer cómo pueden ayudarnos a automatizar nuestro inventario"
            }
        }


class AppointmentResponse(BaseModel):
    """
    Schema para la respuesta de una cita.
    """
    id: int
    lead_id: int
    fecha_preferida: datetime
    servicio_interes: Optional[str]
    estado: str
    created_at: datetime

    class Config:
        from_attributes = True
