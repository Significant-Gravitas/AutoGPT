# NEUS - Guía de Testing

Esta guía proporciona instrucciones detalladas para testear todas las funcionalidades del MVP de NEUS.

## Tabla de Contenidos

1. [Preparación del Entorno](#preparación-del-entorno)
2. [Testing Local (Sin Docker)](#testing-local-sin-docker)
3. [Testing con Docker](#testing-con-docker)
4. [Testing del Backend API](#testing-del-backend-api)
5. [Testing del Frontend](#testing-del-frontend)
6. [Testing de Integración](#testing-de-integración)
7. [Testing del Chatbot](#testing-del-chatbot)
8. [Testing de Base de Datos](#testing-de-base-de-datos)
9. [Checklist de Validación](#checklist-de-validación)
10. [Troubleshooting](#troubleshooting)

---

## Preparación del Entorno

### Requisitos Previos

- Docker y Docker Compose instalados
- Python 3.9+ (para testing local sin Docker)
- Node.js 18+ (para testing local sin Docker)
- PostgreSQL (para testing local sin Docker)
- Una API Key válida de Anthropic Claude
- curl o Postman para testing de API

### Verificar Requisitos

```bash
# Docker
docker --version
# Esperado: Docker version 24.0.x o superior

docker-compose --version
# Esperado: Docker Compose version v2.x o superior

# Python (opcional para testing sin Docker)
python --version
# Esperado: Python 3.9.x o superior

# Node.js (opcional para testing sin Docker)
node --version
# Esperado: v18.x o superior

# PostgreSQL (opcional para testing sin Docker)
psql --version
# Esperado: psql (PostgreSQL) 15.x o superior
```

### Configurar Variables de Entorno

```bash
cd /home/user/neus
cp .env.example .env
```

Editar `.env` y configurar:
```env
# IMPORTANTE: Configurar tu API key real
ANTHROPIC_API_KEY=sk-ant-api03-TU-KEY-AQUI

# Otras variables (generalmente no necesitan cambios)
DATABASE_URL=postgresql://neus:neus_secure_password_123@db:5432/neus
POSTGRES_USER=neus
POSTGRES_PASSWORD=neus_secure_password_123
POSTGRES_DB=neus
CORS_ORIGINS=http://localhost,http://localhost:3000,http://localhost:5173
VITE_API_URL=http://localhost:8000
```

---

## Testing Local (Sin Docker)

### Backend

#### 1. Configurar PostgreSQL

```bash
# Crear base de datos
createdb neus

# O usando psql
psql -U postgres -c "CREATE DATABASE neus;"
```

#### 2. Configurar Backend

```bash
cd /home/user/neus/backend

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tu ANTHROPIC_API_KEY y DATABASE_URL local
# DATABASE_URL=postgresql://postgres:postgres@localhost:5432/neus
```

#### 3. Ejecutar Backend

```bash
uvicorn app.main:app --reload

# Esperado:
# INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
# INFO:     Started reloader process [xxxxx] using WatchFiles
# INFO:     Started server process [xxxxx]
# INFO:     Waiting for application startup.
# INFO:     Application startup complete.
```

#### 4. Verificar Backend

```bash
# En otra terminal
curl http://localhost:8000/api/health

# Esperado:
# {"status":"healthy","service":"NEUS API","version":"1.0.0"}
```

### Frontend

#### 1. Configurar Frontend

```bash
cd /home/user/neus/frontend

# Instalar dependencias
npm install

# Configurar variables de entorno
cp .env.example .env
# Editar .env si es necesario (por defecto usa http://localhost:8000)
```

#### 2. Ejecutar Frontend

```bash
npm run dev

# Esperado:
# VITE v5.x.x  ready in xxx ms
# ➜  Local:   http://localhost:5173/
# ➜  Network: use --host to expose
```

#### 3. Verificar Frontend

Abrir en navegador: http://localhost:5173

Deberías ver la landing page de NEUS.

---

## Testing con Docker

### 1. Deployment Completo

```bash
cd /home/user/neus

# Asegurarse de que .env está configurado
cat .env | grep ANTHROPIC_API_KEY
# Debe mostrar tu API key

# Desplegar
./deploy.sh
```

**Output esperado:**
```
🚀 Desplegando NEUS...
✅ Archivo .env encontrado
✅ Docker está corriendo
🔨 Construyendo imágenes...
✅ Imágenes construidas exitosamente
🚀 Iniciando servicios...
✅ Servicios iniciados
🏥 Verificando salud de los servicios...
✅ Todos los servicios están saludables
✅ NEUS desplegado exitosamente!
```

### 2. Verificar Servicios

```bash
# Ver estado de contenedores
docker-compose ps

# Esperado:
# NAME                SERVICE    STATUS         PORTS
# neus-backend-1      backend    Up (healthy)   0.0.0.0:8000->8000/tcp
# neus-db-1          db         Up (healthy)   5432/tcp
# neus-frontend-1    frontend   Up             0.0.0.0:80->80/tcp
```

### 3. Verificar Logs

```bash
# Ver logs de todos los servicios
./logs.sh

# Ver logs de un servicio específico
./logs.sh backend
./logs.sh frontend
./logs.sh db
```

### 4. Acceder a la Aplicación

- **Frontend:** http://localhost
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## Testing del Backend API

### 1. Health Check

```bash
curl -X GET http://localhost:8000/api/health

# Esperado:
{
  "status": "healthy",
  "service": "NEUS API",
  "version": "1.0.0"
}
```

### 2. Crear Lead

```bash
curl -X POST http://localhost:8000/api/leads \
  -H "Content-Type: application/json" \
  -d '{
    "nombre": "Test Usuario",
    "email": "test@example.com",
    "empresa": "Test Corp",
    "sector": "Retail",
    "mensaje": "Esto es una prueba"
  }'

# Esperado: Status 201
{
  "id": 1,
  "nombre": "Test Usuario",
  "email": "test@example.com",
  "empresa": "Test Corp",
  "sector": "Retail",
  "mensaje": "Esto es una prueba",
  "created_at": "2025-11-01T..."
}
```

### 3. Obtener Lead

```bash
curl -X GET http://localhost:8000/api/leads/1

# Esperado: Status 200
# Retorna el lead con id=1
```

### 4. Crear Appointment

```bash
curl -X POST http://localhost:8000/api/appointments \
  -H "Content-Type: application/json" \
  -d '{
    "nombre": "María García",
    "email": "maria@example.com",
    "empresa": "Tech Inc",
    "sector": "Tecnología",
    "fecha_preferida": "2025-11-15T14:00:00",
    "servicio_interes": "Chatbot Development",
    "mensaje": "Necesito un chatbot para mi empresa"
  }'

# Esperado: Status 201
{
  "id": 1,
  "lead_id": 2,
  "fecha_preferida": "2025-11-15T14:00:00",
  "servicio_interes": "Chatbot Development",
  "estado": "pendiente",
  "created_at": "2025-11-01T...",
  "lead": {
    "id": 2,
    "nombre": "María García",
    "email": "maria@example.com",
    ...
  }
}
```

### 5. Chat con el Bot

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hola, ¿qué servicios ofrecen?",
    "session_id": "test-session-123"
  }'

# Esperado: Status 200
{
  "response": "¡Hola! En NEUS ofrecemos...",
  "session_id": "test-session-123"
}
```

### 6. Obtener Historial de Chat

```bash
curl -X GET http://localhost:8000/api/chat/history/test-session-123

# Esperado: Status 200
{
  "session_id": "test-session-123",
  "messages": [
    {
      "id": 1,
      "message": "Hola, ¿qué servicios ofrecen?",
      "role": "user",
      ...
    },
    {
      "id": 2,
      "message": "¡Hola! En NEUS ofrecemos...",
      "role": "assistant",
      ...
    }
  ]
}
```

### 7. Testing de Errores

```bash
# Email duplicado (debe fallar)
curl -X POST http://localhost:8000/api/leads \
  -H "Content-Type: application/json" \
  -d '{
    "nombre": "Test",
    "email": "test@example.com"
  }'

# Esperado: Status 409 o 400
# Error message apropiado

# Lead no existe
curl -X GET http://localhost:8000/api/leads/999999

# Esperado: Status 404
# {"detail": "Lead not found"}
```

---

## Testing del Frontend

### 1. Navegación General

- [ ] La página carga correctamente
- [ ] Navbar es visible y fijo en la parte superior
- [ ] Scroll suave funciona al hacer click en links del navbar
- [ ] Footer es visible al final de la página

### 2. Hero Section

- [ ] El título principal es visible
- [ ] El subtítulo es visible
- [ ] Los botones CTA son visibles y tienen hover effects
- [ ] Las estadísticas se muestran correctamente

### 3. Sección de Servicios

- [ ] Se muestran 4 servicios en grid
- [ ] Cada servicio tiene ícono, título y descripción
- [ ] En mobile se muestra 1 columna
- [ ] En tablet se muestran 2 columnas
- [ ] En desktop se muestran 2 columnas (grid 2x2)

### 4. Sección de Sectores

- [ ] Se muestran 8 sectores
- [ ] Cada sector tiene ícono y nombre
- [ ] Responsive: 2 cols (mobile), 3 cols (tablet), 4 cols (desktop)
- [ ] Hover effects funcionan

### 5. Sección "Por Qué NEUS"

- [ ] Se muestran 5 razones
- [ ] Cada razón tiene ícono, título y descripción
- [ ] Números ordenados del 1 al 5
- [ ] Diseño responsive

### 6. Formulario de Contacto

**Test Manual:**
1. Llenar todos los campos
2. Hacer click en "Enviar"
3. Verificar mensaje de éxito
4. Verificar en backend que se creó el lead

**Validaciones a probar:**
- [ ] Email inválido muestra error
- [ ] Campos requeridos muestran error si están vacíos
- [ ] Botón se deshabilita durante envío
- [ ] Mensaje de éxito aparece
- [ ] Mensaje de error aparece si falla

### 7. Formulario de Diagnóstico

**Test Manual:**
1. Click en botón "Diagnóstico Gratuito"
2. Modal se abre
3. Llenar formulario
4. Seleccionar fecha futura
5. Enviar

**Validaciones:**
- [ ] Modal abre correctamente
- [ ] Se puede cerrar con X o fuera del modal
- [ ] Date picker funciona
- [ ] No permite fechas pasadas
- [ ] Validación de email funciona
- [ ] Mensaje de éxito aparece

### 8. Chatbot Widget

**Test Manual:**
1. Click en ícono de chat flotante
2. Widget se expande
3. Escribir mensaje
4. Enviar con Enter
5. Ver respuesta del bot

**Validaciones:**
- [ ] Widget flotante es visible
- [ ] Se expande/colapsa correctamente
- [ ] Mensajes se envían correctamente
- [ ] Respuestas del bot aparecen
- [ ] Indicador "escribiendo..." funciona
- [ ] Auto-scroll a nuevos mensajes funciona
- [ ] Session ID persiste en localStorage
- [ ] Al recargar página, sesión se mantiene

### 9. Responsive Design

**Desktop (1920x1080):**
- [ ] Layout se ve bien
- [ ] No hay elementos cortados
- [ ] Imágenes tienen buen tamaño

**Tablet (768x1024):**
- [ ] Grid se adapta (2 columnas generalmente)
- [ ] Navbar se mantiene funcional
- [ ] Texto es legible

**Mobile (375x667):**
- [ ] Todo el contenido es accesible
- [ ] Navbar colapsa apropiadamente
- [ ] Grid cambia a 1 columna
- [ ] Formularios son usables
- [ ] Chatbot no obstruye contenido

### 10. Performance

```bash
# Lighthouse test (en Chrome DevTools)
# 1. Abrir http://localhost en Chrome
# 2. Abrir DevTools (F12)
# 3. Tab "Lighthouse"
# 4. Generar reporte

# Metas:
# - Performance: >80
# - Accessibility: >90
# - Best Practices: >90
# - SEO: >80
```

---

## Testing de Integración

### Flujo Completo 1: Lead Generation

1. **Usuario llena formulario de contacto**
```bash
# En navegador: http://localhost
# Llenar formulario de contacto
# Email: integration-test@example.com
```

2. **Verificar en backend**
```bash
# Ver logs
./logs.sh backend | grep "integration-test"

# O consultar DB directamente
docker-compose exec db psql -U neus -d neus -c "SELECT * FROM leads WHERE email='integration-test@example.com';"

# Esperado: 1 row con los datos
```

### Flujo Completo 2: Appointment Scheduling

1. **Usuario agenda diagnóstico**
```bash
# En navegador: http://localhost
# Click en "Diagnóstico Gratuito"
# Llenar formulario
# Email: appointment-test@example.com
# Fecha: 2025-11-15T14:00:00
```

2. **Verificar en backend**
```bash
# Verificar appointment
docker-compose exec db psql -U neus -d neus -c "SELECT * FROM appointments a JOIN leads l ON a.lead_id = l.id WHERE l.email='appointment-test@example.com';"

# Esperado: 1 row con appointment y lead asociado
```

### Flujo Completo 3: Chatbot Conversation

1. **Usuario chatea con el bot**
```bash
# En navegador: http://localhost
# Abrir widget de chat
# Enviar: "Hola, ¿qué servicios ofrecen?"
# Enviar: "Me interesa el servicio de chatbots"
# Enviar: "¿Cuánto cuesta?"
```

2. **Verificar historial**
```bash
# Obtener session_id del localStorage del navegador
# (abrir DevTools > Application > Local Storage)

# Consultar historial
curl http://localhost:8000/api/chat/history/[SESSION_ID]

# O en DB
docker-compose exec db psql -U neus -d neus -c "SELECT * FROM chat_history ORDER BY created_at DESC LIMIT 10;"

# Esperado: Ver los mensajes del usuario y del bot
```

---

## Testing del Chatbot

### 1. Test de Conocimiento General

Preguntas que el bot debería responder correctamente:

```
Q: "¿Qué servicios ofrecen?"
A: Debería mencionar los 4 pilares (Capacitación, Consultoría, Desarrollo, Infraestructura)

Q: "¿En qué sectores trabajan?"
A: Debería mencionar los 8 sectores (Retail, Salud, Supply Chain, etc.)

Q: "¿Cuál es su propuesta de valor?"
A: Debería mencionar reducción de costos del 40%

Q: "¿Cómo puedo agendar una reunión?"
A: Debería mencionar el formulario de diagnóstico gratuito

Q: "¿Qué es un chatbot?"
A: Debería dar una explicación clara

Q: "¿Pueden ayudarme con automatización?"
A: Debería responder afirmativamente y mencionar servicios relevantes
```

### 2. Test de Contexto

```
# Secuencia de mensajes
1. "Hola"
2. "Soy una empresa de retail"
3. "¿Qué soluciones tienen para mí?"

# El bot debería:
# - Recordar que la empresa es de retail
# - Ofrecer soluciones específicas para retail
# - Mantener el contexto de la conversación
```

### 3. Test de Manejo de Errores

```
Q: [mensaje vacío]
A: Debería manejar gracefully

Q: "asdfasdfasdf" (mensaje sin sentido)
A: Debería pedir clarificación

Q: [mensaje muy largo, 1000+ caracteres]
A: Debería procesar o truncar apropiadamente
```

---

## Testing de Base de Datos

### 1. Verificar Tablas

```bash
docker-compose exec db psql -U neus -d neus

# En psql:
\dt

# Esperado:
#          List of relations
#  Schema |     Name      | Type  | Owner
# --------+---------------+-------+-------
#  public | appointments  | table | neus
#  public | chat_history  | table | neus
#  public | leads         | table | neus
```

### 2. Verificar Estructura de Tablas

```sql
-- Leads
\d leads

-- Esperado:
-- Columnas: id, nombre, email (unique), empresa, sector, mensaje, created_at

-- Appointments
\d appointments

-- Esperado:
-- Columnas: id, lead_id (FK), fecha_preferida, servicio_interes, estado, created_at

-- Chat History
\d chat_history

-- Esperado:
-- Columnas: id, session_id, message, role, created_at
```

### 3. Verificar Relaciones

```sql
-- Verificar foreign key constraint
SELECT * FROM appointments a
JOIN leads l ON a.lead_id = l.id
LIMIT 5;

-- Debería mostrar appointments con sus leads asociados
```

### 4. Verificar Índices

```sql
-- Ver índices
\di

-- Debería haber índices en:
-- - leads.email (unique)
-- - appointments.lead_id (FK index)
```

### 5. Testing de Constraints

```sql
-- Email único (debe fallar)
INSERT INTO leads (nombre, email) VALUES ('Test', 'duplicate@test.com');
INSERT INTO leads (nombre, email) VALUES ('Test2', 'duplicate@test.com');
-- Esperado: ERROR: duplicate key value violates unique constraint

-- NOT NULL (debe fallar)
INSERT INTO leads (nombre) VALUES ('Test');
-- Esperado: ERROR: null value in column "email" violates not-null constraint
```

---

## Checklist de Validación

### Pre-Deployment

- [ ] `.env` configurado con API key real
- [ ] Docker y Docker Compose instalados
- [ ] Puertos 80, 8000, 5432 disponibles

### Post-Deployment

**Infraestructura:**
- [ ] Todos los contenedores corriendo (`docker-compose ps`)
- [ ] Health checks pasando
- [ ] Logs no muestran errores críticos

**Backend:**
- [ ] Health check responde OK
- [ ] API docs accesibles en /docs
- [ ] Todos los endpoints responden
- [ ] Database tiene las 3 tablas

**Frontend:**
- [ ] Landing page carga completamente
- [ ] No hay errores en consola del navegador
- [ ] Todos los componentes visibles
- [ ] Responsive en mobile/tablet/desktop

**Funcionalidades:**
- [ ] Formulario de contacto crea lead
- [ ] Formulario de diagnóstico crea appointment
- [ ] Chatbot responde correctamente
- [ ] Session de chat persiste
- [ ] Datos se guardan en DB

**Performance:**
- [ ] Frontend carga en <3 segundos
- [ ] API responde en <500ms (sin chatbot)
- [ ] Chatbot responde en <5 segundos
- [ ] No hay memory leaks visibles

**Security:**
- [ ] API key no está expuesta en frontend
- [ ] CORS configurado apropiadamente
- [ ] No hay SQL injection vulnerabilities
- [ ] Variables de entorno usadas correctamente

---

## Troubleshooting

### Backend no inicia

```bash
# Ver logs
./logs.sh backend

# Problemas comunes:
# 1. PostgreSQL no está ready
#    Solución: Esperar unos segundos más

# 2. ANTHROPIC_API_KEY no configurada
#    Solución: Verificar .env

# 3. Puerto 8000 ocupado
#    Solución: Cambiar puerto en docker-compose.yml
```

### Frontend no carga

```bash
# Ver logs
./logs.sh frontend

# Problemas comunes:
# 1. Backend no está corriendo
#    Solución: Verificar backend primero

# 2. VITE_API_URL incorrecta
#    Solución: Verificar .env.example del frontend

# 3. Puerto 80 ocupado
#    Solución: Cambiar puerto en docker-compose.yml
```

### Chatbot no responde

```bash
# Verificar logs del backend
./logs.sh backend | grep -i "anthropic\|claude\|api"

# Problemas comunes:
# 1. API key inválida
#    Solución: Verificar ANTHROPIC_API_KEY en .env

# 2. Rate limit excedido
#    Solución: Esperar o usar otra API key

# 3. Error de red
#    Solución: Verificar conectividad a internet
```

### Base de datos no persiste datos

```bash
# Verificar volumen
docker volume ls | grep neus

# Debería existir: neus_postgres_data

# Verificar que el volumen está montado
docker-compose exec db df -h

# Si los datos se pierden al reiniciar:
# - Verificar que el volumen está declarado en docker-compose.yml
# - No usar 'docker-compose down -v' (borra volúmenes)
```

---

## Comandos Útiles para Testing

```bash
# Reconstruir todo desde cero
docker-compose down -v
docker-compose build --no-cache
./deploy.sh

# Reiniciar un servicio específico
docker-compose restart backend

# Ver recursos usados
docker stats

# Conectar a PostgreSQL
docker-compose exec db psql -U neus -d neus

# Ejecutar comando en contenedor
docker-compose exec backend bash

# Ver variables de entorno de un servicio
docker-compose exec backend env | grep ANTHROPIC

# Seguir logs en tiempo real
docker-compose logs -f backend

# Limpiar todo (CUIDADO: borra datos)
docker-compose down -v
docker system prune -a
```

---

## Testing de Producción

Para testing en un entorno de producción/staging:

1. Seguir guía en [DEPLOYMENT.md](DEPLOYMENT.md)
2. Configurar dominio de prueba
3. Configurar HTTPS con Let's Encrypt
4. Ejecutar todos los tests de este documento
5. Verificar logs de producción
6. Monitorear rendimiento

---

**Última Actualización:** 2025-11-01
**Versión:** 1.0.0
