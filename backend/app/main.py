from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

from app.database import create_tables
from app.routes import leads, appointments, chat

# Cargar variables de entorno
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager para inicializar recursos al arrancar la app.
    """
    # Startup: Crear tablas en la base de datos
    print("🚀 Iniciando aplicación NEUS Backend...")
    print("📊 Creando tablas en la base de datos...")
    create_tables()
    print("✅ Tablas creadas correctamente")

    yield

    # Shutdown: Aquí se pueden cerrar conexiones si es necesario
    print("👋 Cerrando aplicación NEUS Backend...")


# Crear aplicación FastAPI
app = FastAPI(
    title="NEUS API",
    description="API REST para la plataforma NEUS - Servicios de IA Empresarial",
    version="1.0.0",
    lifespan=lifespan,
)

# Configurar CORS
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(leads.router)
app.include_router(appointments.router)
app.include_router(chat.router)


# Health check endpoint
@app.get("/api/health", tags=["Health"])
def health_check():
    """
    Endpoint de health check para verificar que el API está funcionando.
    """
    return {
        "status": "healthy",
        "service": "NEUS API",
        "version": "1.0.0",
    }


# Root endpoint
@app.get("/", tags=["Root"])
def root():
    """
    Endpoint raíz con información básica del API.
    """
    return {
        "message": "Bienvenido a NEUS API",
        "description": "API REST para servicios de IA empresarial",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
