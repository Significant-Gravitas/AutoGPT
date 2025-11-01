from pydantic import BaseModel, Field
from typing import Optional


class ChatMessage(BaseModel):
    """
    Schema para un mensaje de chat.
    """
    message: str = Field(..., min_length=1, description="Mensaje del usuario")
    session_id: Optional[str] = Field(None, description="ID de sesión para mantener contexto")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "¿Qué servicios ofrecen?",
                "session_id": "abc123-session"
            }
        }


class ChatResponse(BaseModel):
    """
    Schema para la respuesta del chatbot.
    """
    response: str = Field(..., description="Respuesta del chatbot")
    session_id: str = Field(..., description="ID de sesión")

    class Config:
        json_schema_extra = {
            "example": {
                "response": "NEUS ofrece 4 servicios principales: Capacitación en IA, Consultoría Estratégica, Desarrollo y Automatización, e Infraestructura y Seguridad.",
                "session_id": "abc123-session"
            }
        }
