# NEUS Backend API

Backend API REST para la plataforma NEUS - Servicios de IA Empresarial.

## Tecnologías

- **FastAPI**: Framework web moderno y rápido
- **SQLAlchemy**: ORM para PostgreSQL
- **PostgreSQL**: Base de datos relacional
- **Anthropic Claude**: API de IA para el chatbot
- **Docker**: Containerización

## Estructura del Proyecto

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # Aplicación FastAPI principal
│   ├── database.py          # Configuración de base de datos
│   ├── models/              # Modelos SQLAlchemy
│   │   ├── lead.py
│   │   ├── appointment.py
│   │   └── chat_history.py
│   ├── schemas/             # Schemas Pydantic
│   │   ├── lead.py
│   │   ├── appointment.py
│   │   └── chat.py
│   ├── routes/              # Endpoints del API
│   │   ├── leads.py
│   │   ├── appointments.py
│   │   └── chat.py
│   └── services/            # Servicios externos
│       └── chatbot.py       # Integración con Claude
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

## Instalación y Configuración

### 1. Clonar el repositorio y navegar al directorio backend

```bash
cd /home/user/neus/backend
```

### 2. Crear y activar entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # En Linux/Mac
# o
venv\Scripts\activate  # En Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

Copiar el archivo `.env.example` a `.env` y configurar las variables:

```bash
cp .env.example .env
```

Editar `.env` con tus credenciales:

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/neus
ANTHROPIC_API_KEY=sk-ant-api03-tu-key-aqui
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

### 5. Configurar PostgreSQL

Asegúrate de tener PostgreSQL instalado y corriendo. Crear la base de datos:

```bash
createdb neus
# o usando psql:
psql -U postgres -c "CREATE DATABASE neus;"
```

### 6. Ejecutar la aplicación

```bash
uvicorn app.main:app --reload
```

El API estará disponible en: `http://localhost:8000`

- Documentación interactiva (Swagger): `http://localhost:8000/docs`
- Documentación alternativa (ReDoc): `http://localhost:8000/redoc`

## Endpoints Disponibles

### Health Check
- `GET /api/health` - Verificar estado del servicio

### Leads
- `POST /api/leads` - Crear nuevo lead
- `GET /api/leads/{lead_id}` - Obtener lead por ID

### Appointments (Citas)
- `POST /api/appointments` - Crear nueva cita de diagnóstico
- `GET /api/appointments/{appointment_id}` - Obtener cita por ID

### Chat
- `POST /api/chat` - Enviar mensaje al chatbot
- `GET /api/chat/history/{session_id}` - Obtener historial de chat

## Uso con Docker

### Construir la imagen

```bash
docker build -t neus-backend .
```

### Ejecutar el contenedor

```bash
docker run -p 8000:8000 --env-file .env neus-backend
```

## Ejemplos de Uso

### Crear un Lead

```bash
curl -X POST "http://localhost:8000/api/leads" \
  -H "Content-Type: application/json" \
  -d '{
    "nombre": "Juan Pérez",
    "email": "juan@empresa.com",
    "empresa": "Empresa XYZ",
    "sector": "Retail",
    "mensaje": "Me interesa automatización"
  }'
```

### Agendar una Cita

```bash
curl -X POST "http://localhost:8000/api/appointments" \
  -H "Content-Type: application/json" \
  -d '{
    "nombre": "María García",
    "email": "maria@empresa.com",
    "empresa": "Retail Corp",
    "sector": "Retail",
    "fecha_preferida": "2025-11-15T10:00:00",
    "servicio_interes": "Automatización de procesos"
  }'
```

### Chat con el Bot

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "¿Qué servicios ofrecen?",
    "session_id": "test-session-123"
  }'
```

## Desarrollo

### Activar modo debug

En `app/database.py`, cambiar `echo=False` a `echo=True` para ver las queries SQL.

### Crear migración de base de datos

Las tablas se crean automáticamente al iniciar la aplicación. Para migraciones más complejas, se recomienda usar Alembic (no incluido en este MVP).

## Notas

- El chatbot requiere una API key válida de Anthropic Claude
- La base de datos PostgreSQL debe estar corriendo antes de iniciar el backend
- Los endpoints están documentados automáticamente en `/docs`

## Problemas Comunes

### Error de conexión a PostgreSQL
- Verificar que PostgreSQL está corriendo: `pg_isready`
- Verificar credenciales en DATABASE_URL

### Error de API Key de Anthropic
- Verificar que ANTHROPIC_API_KEY está configurada en .env
- Verificar que la key es válida y tiene formato `sk-ant-api03-...`

### CORS Error
- Verificar que el frontend está en la lista de CORS_ORIGINS
- Por defecto permite localhost:3000 y localhost:5173
