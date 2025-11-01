from sqlalchemy import Column, Integer, String, Text, DateTime
from datetime import datetime
from app.database import Base


class ChatHistory(Base):
    """
    Modelo para almacenar el historial de conversaciones del chatbot.
    """
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    message = Column(Text, nullable=False)
    role = Column(String(20), nullable=False)  # user o assistant
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<ChatHistory {self.session_id} - {self.role}>"
