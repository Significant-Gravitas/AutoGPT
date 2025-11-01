# NEUS - Plataforma de Servicios de IA Empresarial

> Impulsando la eficiencia empresarial con Inteligencia Artificial

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/react-18+-61dafb.svg)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/typescript-5+-blue.svg)](https://www.typescriptlang.org/)

## Descripción

NEUS es una consultora especializada en impulsar la eficiencia empresarial mediante Inteligencia Artificial. Este MVP (Minimum Viable Product) es una plataforma web moderna que permite:

- Presentar nuestros servicios de IA empresarial
- Captar leads de potenciales clientes
- Ofrecer diagnósticos gratuitos
- Demostrar capacidades con un chatbot inteligente

### Propuesta de Valor

**Reducir costos operativos hasta 40% mediante automatización inteligente**

## Características

- Landing page moderna y responsive
- Formulario de contacto para captura de leads
- Sistema de agendamiento de diagnósticos gratuitos
- Chatbot demo integrado con IA (Anthropic Claude)
- API REST completa con documentación interactiva
- Base de datos PostgreSQL para almacenamiento persistente
- Arquitectura containerizada con Docker

## Arquitectura

### Stack Tecnológico

- **Frontend**: React 18 + TypeScript + Vite + Tailwind CSS
- **Backend**: Python 3.11 + FastAPI + SQLAlchemy
- **Base de Datos**: PostgreSQL 15
- **IA**: Anthropic Claude API (claude-3-5-sonnet)
- **Deployment**: Docker + Docker Compose
- **Web Server**: Nginx (producción)

### Diagrama de Arquitectura

```
┌─────────────┐     HTTP      ┌──────────────┐     SQL       ┌──────────────┐
│   Frontend  │ ───────────► │   Backend    │ ───────────► │  PostgreSQL  │
│ React + TS  │   REST API    │   FastAPI    │   SQLAlchemy  │   Database   │
│  (Nginx)    │ ◄─────────── │   Python     │ ◄─────────── │              │
└─────────────┘               └──────────────┘               └──────────────┘
                                     │
                                     │ API Calls
                                     ▼
                              ┌──────────────┐
                              │  Anthropic   │
                              │  Claude API  │
                              └──────────────┘
```

## Requisitos Previos

Antes de comenzar, asegúrate de tener instalado:

- **Docker** (versión 20.10+) - [Instalar Docker](https://docs.docker.com/get-docker/)
- **Docker Compose** (versión 2.0+) - [Instalar Docker Compose](https://docs.docker.com/compose/install/)
- **Anthropic API Key** - [Obtener API Key](https://console.anthropic.com/)

### Verificar instalación

```bash
docker --version
docker-compose --version
```

## Instalación Rápida (Quick Start)

1. **Clonar el repositorio** (o navegar al directorio del proyecto)

```bash
cd /home/user/neus
```

2. **Configurar variables de entorno**

```bash
cp .env.example .env
```

Edita el archivo `.env` y configura:
- `DB_PASSWORD`: Una contraseña segura para PostgreSQL
- `ANTHROPIC_API_KEY`: Tu API key de Anthropic

3. **Ejecutar el deployment**

```bash
./deploy.sh
```

4. **Acceder a la aplicación**

- Frontend: http://localhost
- Backend API: http://localhost:8000
- Documentación API: http://localhost:8000/docs

## Instalación Detallada

### 1. Configuración de Variables de Entorno

El archivo `.env` debe contener las siguientes variables:

```env
# Database
DB_PASSWORD=your-secure-password-here

# Backend API Keys
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

# Production (opcional)
PRODUCTION_URL=https://neus.example.com
```

### 2. Construcción de Imágenes Docker

```bash
# Construir todas las imágenes
docker-compose build

# Construir solo un servicio específico
docker-compose build backend
docker-compose build frontend
```

### 3. Iniciar Servicios

```bash
# Iniciar todos los servicios en background
docker-compose up -d

# Iniciar con logs visibles
docker-compose up

# Iniciar un servicio específico
docker-compose up -d backend
```

### 4. Verificar Estado de Servicios

```bash
# Ver estado de contenedores
docker-compose ps

# Ver logs
./logs.sh

# Ver logs de un servicio específico
./logs.sh backend
./logs.sh frontend
./logs.sh db
```

## Estructura del Proyecto

```
/home/user/neus/
├── backend/                    # API Backend (FastAPI)
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py            # Aplicación principal
│   │   ├── database.py        # Configuración DB
│   │   ├── models/            # Modelos SQLAlchemy
│   │   ├── schemas/           # Schemas Pydantic
│   │   ├── routes/            # Endpoints API
│   │   └── services/          # Lógica de negocio
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── .env.example
│   └── README.md
│
├── frontend/                   # Frontend (React + TypeScript)
│   ├── src/
│   │   ├── components/        # Componentes React
│   │   ├── services/          # API calls
│   │   ├── types/             # TypeScript types
│   │   ├── App.tsx           # Componente principal
│   │   └── main.tsx          # Entry point
│   ├── package.json
│   ├── Dockerfile
│   ├── .env.example
│   └── README.md
│
├── docker-compose.yml          # Orquestación de servicios
├── .env                       # Variables de entorno (no versionado)
├── .env.example               # Template de variables
├── deploy.sh                  # Script de deployment
├── stop.sh                    # Script para detener servicios
├── logs.sh                    # Script para ver logs
├── NEUS-README.md            # Este archivo
├── DEPLOYMENT.md              # Guía de deployment a producción
└── MVP_CONTEXT.md             # Contexto del desarrollo
```

## Uso

### Frontend

El frontend está disponible en http://localhost y ofrece:

1. **Hero Section**: Propuesta de valor y CTAs principales
2. **Servicios**: 4 pilares de servicios (Capacitación, Consultoría, Desarrollo, Seguridad)
3. **Sectores**: 8 sectores objetivo
4. **Por qué NEUS**: 5 razones para elegir NEUS
5. **Formulario de Contacto**: Captura de leads
6. **Diagnóstico Gratuito**: Modal para agendar consultas
7. **Chatbot**: Widget flotante con IA para responder preguntas

### Backend API

La API REST está disponible en http://localhost:8000 con los siguientes endpoints:

#### Health Check
```bash
GET /api/health
```

#### Leads
```bash
# Crear lead
POST /api/leads
Content-Type: application/json

{
  "nombre": "Juan Pérez",
  "email": "juan@empresa.com",
  "empresa": "Empresa SA",
  "sector": "Retail",
  "mensaje": "Quiero información sobre automatización"
}

# Obtener lead
GET /api/leads/{lead_id}
```

#### Appointments
```bash
# Crear cita de diagnóstico
POST /api/appointments
Content-Type: application/json

{
  "nombre": "María García",
  "email": "maria@empresa.com",
  "empresa": "Tech Corp",
  "sector": "Salud",
  "fecha_preferida": "2025-11-15",
  "servicio_interes": "Consultoría Estratégica",
  "mensaje": "Necesito optimizar procesos"
}

# Obtener appointment
GET /api/appointments/{appointment_id}
```

#### Chat
```bash
# Enviar mensaje al chatbot
POST /api/chat
Content-Type: application/json

{
  "message": "¿Qué servicios ofrecen?",
  "session_id": "uuid-here"  # Opcional
}

# Obtener historial
GET /api/chat/history/{session_id}
```

### Documentación Interactiva

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Comandos Útiles

### Docker Compose

```bash
# Iniciar todos los servicios
docker-compose up -d

# Detener todos los servicios
docker-compose down

# Detener y eliminar volúmenes (CUIDADO: elimina la base de datos)
docker-compose down -v

# Reiniciar un servicio
docker-compose restart backend

# Ver logs en tiempo real
docker-compose logs -f

# Ver logs de un servicio específico
docker-compose logs -f backend

# Ejecutar comando en un contenedor
docker-compose exec backend bash
docker-compose exec db psql -U neus -d neus

# Reconstruir imágenes
docker-compose build --no-cache
```

### Scripts Helper

```bash
# Deployment completo
./deploy.sh

# Detener servicios
./stop.sh

# Ver logs (todos los servicios)
./logs.sh

# Ver logs de un servicio específico
./logs.sh backend
./logs.sh frontend
./logs.sh db
```

### Base de Datos

```bash
# Conectar a PostgreSQL
docker-compose exec db psql -U neus -d neus

# Backup de base de datos
docker-compose exec db pg_dump -U neus neus > backup.sql

# Restaurar base de datos
docker-compose exec -T db psql -U neus -d neus < backup.sql

# Ver tablas
docker-compose exec db psql -U neus -d neus -c "\dt"

# Ver contenido de tabla
docker-compose exec db psql -U neus -d neus -c "SELECT * FROM leads;"
```

## Desarrollo Local

### Backend (sin Docker)

```bash
cd /home/user/neus/backend

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar .env
cp .env.example .env
# Editar .env con tus credenciales

# Ejecutar servidor de desarrollo
uvicorn app.main:app --reload

# La API estará en http://localhost:8000
```

### Frontend (sin Docker)

```bash
cd /home/user/neus/frontend

# Instalar dependencias
npm install

# Configurar .env
cp .env.example .env
# Verificar que VITE_API_URL=http://localhost:8000

# Ejecutar servidor de desarrollo
npm run dev

# El frontend estará en http://localhost:5173
```

## Troubleshooting

### Error: "Cannot connect to database"

**Problema**: El backend no puede conectarse a PostgreSQL

**Solución**:
```bash
# Verificar que PostgreSQL está corriendo y healthy
docker-compose ps

# Verificar logs de la base de datos
docker-compose logs db

# Reiniciar el servicio de base de datos
docker-compose restart db

# Si persiste, reconstruir el contenedor
docker-compose down
docker-compose up -d
```

### Error: "Port already in use"

**Problema**: El puerto 80, 8000 o 5432 ya está en uso

**Solución**:
```bash
# Ver qué proceso usa el puerto
sudo lsof -i :80
sudo lsof -i :8000
sudo lsof -i :5432

# Detener el proceso o cambiar el puerto en docker-compose.yml
# Ejemplo para cambiar puerto del frontend:
# ports:
#   - "8080:80"  # Usar puerto 8080 en lugar de 80
```

### Error: "Anthropic API Key invalid"

**Problema**: La API key de Anthropic no es válida

**Solución**:
```bash
# Verificar que la API key está configurada en .env
cat .env | grep ANTHROPIC_API_KEY

# Debe tener formato: sk-ant-api03-...
# Obtener nueva key en: https://console.anthropic.com/

# Reiniciar backend después de actualizar
docker-compose restart backend
```

### Frontend muestra error de conexión

**Problema**: El frontend no puede conectarse al backend

**Solución**:
```bash
# Verificar que el backend está corriendo
docker-compose ps backend

# Verificar logs del backend
docker-compose logs backend

# Verificar que CORS está configurado correctamente
# En backend/.env debe tener:
# CORS_ORIGINS=http://localhost,http://localhost:80

# Reiniciar servicios
docker-compose restart backend frontend
```

### Chatbot no responde

**Problema**: El chatbot no genera respuestas

**Solución**:
1. Verificar API key de Anthropic en `.env`
2. Verificar límites de API (quota, rate limits)
3. Ver logs del backend: `docker-compose logs backend`
4. Verificar conectividad a internet del contenedor

### Limpiar todo y empezar de nuevo

```bash
# Detener todos los servicios
docker-compose down

# Eliminar volúmenes (CUIDADO: elimina la base de datos)
docker-compose down -v

# Eliminar imágenes
docker-compose down --rmi all

# Reconstruir todo
docker-compose build --no-cache

# Iniciar de nuevo
./deploy.sh
```

## Deployment a Producción

Para deployment a producción (VPS, cloud, etc.), consulta la guía detallada en:

**[DEPLOYMENT.md](DEPLOYMENT.md)**

Incluye:
- Configuración de VPS (DigitalOcean, AWS, etc.)
- Configuración de dominio y DNS
- HTTPS con Let's Encrypt
- Nginx como reverse proxy
- Firewall y seguridad
- Backup y monitoreo

## Variables de Entorno

### Backend

| Variable | Descripción | Requerido | Ejemplo |
|----------|-------------|-----------|---------|
| `DATABASE_URL` | URL de PostgreSQL | Sí | `postgresql://user:pass@db:5432/neus` |
| `ANTHROPIC_API_KEY` | API key de Anthropic | Sí | `sk-ant-api03-...` |
| `CORS_ORIGINS` | Orígenes permitidos | No | `http://localhost,http://localhost:80` |

### Frontend

| Variable | Descripción | Requerido | Ejemplo |
|----------|-------------|-----------|---------|
| `VITE_API_URL` | URL del backend API | Sí | `http://localhost:8000` |

### Docker Compose

| Variable | Descripción | Requerido | Ejemplo |
|----------|-------------|-----------|---------|
| `DB_PASSWORD` | Contraseña de PostgreSQL | Sí | `secure_password_123` |
| `ANTHROPIC_API_KEY` | API key de Anthropic | Sí | `sk-ant-api03-...` |

## Testing

### Testing Manual

1. **Frontend**:
   - Navega a http://localhost
   - Verifica que todas las secciones cargan correctamente
   - Prueba el formulario de contacto
   - Prueba el formulario de diagnóstico
   - Interactúa con el chatbot

2. **Backend**:
   - Navega a http://localhost:8000/docs
   - Prueba cada endpoint desde Swagger UI
   - Verifica respuestas y códigos de estado
   - Prueba validaciones de datos

3. **Base de Datos**:
   ```bash
   docker-compose exec db psql -U neus -d neus -c "SELECT * FROM leads;"
   docker-compose exec db psql -U neus -d neus -c "SELECT * FROM appointments;"
   docker-compose exec db psql -U neus -d neus -c "SELECT * FROM chat_history;"
   ```

### Testing Automatizado

```bash
# Backend tests (si se implementan)
cd backend
pytest

# Frontend tests (si se implementan)
cd frontend
npm test
```

## Licencia

Este proyecto está bajo la licencia MIT. Ver archivo [LICENSE](LICENSE) para más detalles.

## Contacto y Soporte

- **Documentación Backend**: [backend/README.md](backend/README.md)
- **Documentación Frontend**: [frontend/README.md](frontend/README.md)
- **Guía de Deployment**: [DEPLOYMENT.md](DEPLOYMENT.md)
- **Contexto del MVP**: [MVP_CONTEXT.md](MVP_CONTEXT.md)

---

**Desarrollado con por el equipo de NEUS** | 2025
