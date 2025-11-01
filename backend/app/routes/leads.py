from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from app.database import get_db
from app.models.lead import Lead
from app.schemas.lead import LeadCreate, LeadResponse

router = APIRouter(prefix="/api/leads", tags=["Leads"])


@router.post("/", response_model=LeadResponse, status_code=status.HTTP_201_CREATED)
def create_lead(lead_data: LeadCreate, db: Session = Depends(get_db)):
    """
    Crea un nuevo lead en la base de datos.

    - **nombre**: Nombre completo del lead (requerido)
    - **email**: Email del lead (requerido, único)
    - **empresa**: Nombre de la empresa (opcional)
    - **sector**: Sector empresarial (opcional)
    - **mensaje**: Mensaje o consulta (opcional)
    """
    try:
        # Verificar si el email ya existe
        existing_lead = db.query(Lead).filter(Lead.email == lead_data.email).first()
        if existing_lead:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Ya existe un lead con el email {lead_data.email}"
            )

        # Crear nuevo lead
        new_lead = Lead(
            nombre=lead_data.nombre,
            email=lead_data.email,
            empresa=lead_data.empresa,
            sector=lead_data.sector,
            mensaje=lead_data.mensaje,
        )

        db.add(new_lead)
        db.commit()
        db.refresh(new_lead)

        return new_lead

    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Error de integridad en la base de datos. El email podría estar duplicado."
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al crear el lead: {str(e)}"
        )


@router.get("/{lead_id}", response_model=LeadResponse)
def get_lead(lead_id: int, db: Session = Depends(get_db)):
    """
    Obtiene un lead por su ID.
    """
    lead = db.query(Lead).filter(Lead.id == lead_id).first()
    if not lead:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lead con ID {lead_id} no encontrado"
        )
    return lead
