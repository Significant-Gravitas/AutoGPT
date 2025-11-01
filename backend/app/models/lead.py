from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base


class Lead(Base):
    """
    Modelo para almacenar información de leads (clientes potenciales).
    """
    __tablename__ = "leads"

    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False, unique=True, index=True)
    empresa = Column(String(255), nullable=True)
    sector = Column(String(100), nullable=True)
    mensaje = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relación con appointments
    appointments = relationship("Appointment", back_populates="lead", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Lead {self.nombre} ({self.email})>"
