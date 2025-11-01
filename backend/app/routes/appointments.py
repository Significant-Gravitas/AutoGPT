from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.lead import Lead
from app.models.appointment import Appointment
from app.schemas.appointment import AppointmentCreate, AppointmentResponse

router = APIRouter(prefix="/api/appointments", tags=["Appointments"])


@router.post("/", response_model=AppointmentResponse, status_code=status.HTTP_201_CREATED)
def create_appointment(appointment_data: AppointmentCreate, db: Session = Depends(get_db)):
    """
    Crea una nueva cita de diagnóstico gratuito.

    Si el lead (email) no existe, lo crea automáticamente.
    Si ya existe, usa el lead existente.

    - **nombre**: Nombre completo
    - **email**: Email (se usa para identificar o crear el lead)
    - **empresa**: Nombre de la empresa (opcional)
    - **sector**: Sector empresarial (opcional)
    - **fecha_preferida**: Fecha y hora preferida para la cita
    - **servicio_interes**: Servicio de interés (opcional)
    - **mensaje**: Mensaje adicional (opcional)
    """
    try:
        # Buscar o crear el lead
        lead = db.query(Lead).filter(Lead.email == appointment_data.email).first()

        if not lead:
            # Crear nuevo lead si no existe
            lead = Lead(
                nombre=appointment_data.nombre,
                email=appointment_data.email,
                empresa=appointment_data.empresa,
                sector=appointment_data.sector,
                mensaje=appointment_data.mensaje,
            )
            db.add(lead)
            db.flush()  # Para obtener el ID sin hacer commit

        # Crear la cita
        new_appointment = Appointment(
            lead_id=lead.id,
            fecha_preferida=appointment_data.fecha_preferida,
            servicio_interes=appointment_data.servicio_interes,
            estado="pendiente",
        )

        db.add(new_appointment)
        db.commit()
        db.refresh(new_appointment)

        return new_appointment

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al crear la cita: {str(e)}"
        )


@router.get("/{appointment_id}", response_model=AppointmentResponse)
def get_appointment(appointment_id: int, db: Session = Depends(get_db)):
    """
    Obtiene una cita por su ID.
    """
    appointment = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if not appointment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cita con ID {appointment_id} no encontrada"
        )
    return appointment
