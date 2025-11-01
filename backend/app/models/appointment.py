from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base


class Appointment(Base):
    """
    Modelo para almacenar citas de diagnóstico gratuito agendadas.
    """
    __tablename__ = "appointments"

    id = Column(Integer, primary_key=True, index=True)
    lead_id = Column(Integer, ForeignKey("leads.id"), nullable=False)
    fecha_preferida = Column(DateTime, nullable=False)
    servicio_interes = Column(String(255), nullable=True)
    estado = Column(String(50), default="pendiente", nullable=False)  # pendiente, confirmado, completado, cancelado
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relación con lead
    lead = relationship("Lead", back_populates="appointments")

    def __repr__(self):
        return f"<Appointment {self.id} - Lead {self.lead_id} - {self.estado}>"
