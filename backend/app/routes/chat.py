from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.chat_history import ChatHistory
from app.schemas.chat import ChatMessage, ChatResponse
from app.services.chatbot import get_chatbot_response
import uuid

router = APIRouter(prefix="/api/chat", tags=["Chat"])


@router.post("/", response_model=ChatResponse)
async def chat(message_data: ChatMessage, db: Session = Depends(get_db)):
    """
    Endpoint para el chatbot de NEUS.

    Recibe un mensaje del usuario y devuelve una respuesta del chatbot.
    Mantiene el historial de conversaciones en la base de datos.

    - **message**: Mensaje del usuario (requerido)
    - **session_id**: ID de sesión para mantener contexto (opcional, se genera si no se proporciona)
    """
    try:
        # Generar session_id si no se proporciona
        session_id = message_data.session_id or str(uuid.uuid4())

        # Guardar mensaje del usuario en el historial
        user_message = ChatHistory(
            session_id=session_id,
            message=message_data.message,
            role="user",
        )
        db.add(user_message)
        db.commit()

        # Obtener respuesta del chatbot
        bot_response = await get_chatbot_response(
            message=message_data.message,
            session_id=session_id
        )

        # Guardar respuesta del bot en el historial
        assistant_message = ChatHistory(
            session_id=session_id,
            message=bot_response,
            role="assistant",
        )
        db.add(assistant_message)
        db.commit()

        return ChatResponse(
            response=bot_response,
            session_id=session_id
        )

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en el servicio de chat: {str(e)}"
        )


@router.get("/history/{session_id}")
def get_chat_history(session_id: str, db: Session = Depends(get_db)):
    """
    Obtiene el historial de chat de una sesión específica.

    Útil para debugging o para mostrar conversaciones previas.
    """
    history = (
        db.query(ChatHistory)
        .filter(ChatHistory.session_id == session_id)
        .order_by(ChatHistory.created_at.asc())
        .all()
    )

    if not history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No se encontró historial para la sesión {session_id}"
        )

    return [
        {
            "message": msg.message,
            "role": msg.role,
            "created_at": msg.created_at,
        }
        for msg in history
    ]
