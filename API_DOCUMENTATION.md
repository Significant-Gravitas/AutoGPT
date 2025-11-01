# NEUS API - Documentación Completa

## Información General

**Base URL:** `http://localhost:8000` (desarrollo) o `https://api.neus.com` (producción)

**Versión:** 1.0.0

**Tipo:** REST API

**Formato de datos:** JSON

**Autenticación:** No requerida en MVP (a implementar en futuras versiones)

## Endpoints Disponibles

### Tabla de Contenidos
1. [Health Check](#1-health-check)
2. [Información del API](#2-información-del-api)
3. [Leads - Crear](#3-crear-lead)
4. [Leads - Obtener](#4-obtener-lead)
5. [Appointments - Crear](#5-crear-appointment)
6. [Appointments - Obtener](#6-obtener-appointment)
7. [Chat - Enviar Mensaje](#7-enviar-mensaje-al-chatbot)
8. [Chat - Historial](#8-obtener-historial-de-chat)

---

## 1. Health Check

Verifica el estado del servicio.

### Request

```http
GET /api/health
```

### Response

**Status Code:** `200 OK`

```json
{
  "status": "healthy",
  "service": "NEUS API",
  "version": "1.0.0"
}
```

### Ejemplo curl

```bash
curl -X GET "http://localhost:8000/api/health"
```

### Ejemplo JavaScript (fetch)

```javascript
const response = await fetch('http://localhost:8000/api/health');
const data = await response.json();
console.log(data);
```

---

## 2. Información del API

Obtiene información general del API.

### Request

```http
GET /
```

### Response

**Status Code:** `200 OK`

```json
{
  "name": "NEUS API",
  "version": "1.0.0",
  "description": "API REST para NEUS - Plataforma de Servicios de IA Empresarial"
}
```

### Ejemplo curl

```bash
curl -X GET "http://localhost:8000/"
```

---

## 3. Crear Lead

Crea un nuevo lead (contacto) en el sistema.

### Request

```http
POST /api/leads
Content-Type: application/json
```

**Body Parameters:**

| Campo | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| `nombre` | string | ✅ Sí | Nombre completo del lead |
| `email` | string | ✅ Sí | Email (debe ser válido y único) |
| `empresa` | string | ❌ No | Nombre de la empresa |
| `sector` | string | ❌ No | Sector de la empresa (ej: "Retail", "Salud") |
| `mensaje` | string | ❌ No | Mensaje o consulta del lead |

**Request Body:**

```json
{
  "nombre": "Juan Pérez",
  "email": "juan.perez@empresa.com",
  "empresa": "Retail Solutions SA",
  "sector": "Retail",
  "mensaje": "Me interesa conocer más sobre automatización de procesos"
}
```

### Response

**Status Code:** `201 Created`

```json
{
  "id": 1,
  "nombre": "Juan Pérez",
  "email": "juan.perez@empresa.com",
  "empresa": "Retail Solutions SA",
  "sector": "Retail",
  "mensaje": "Me interesa conocer más sobre automatización de procesos",
  "created_at": "2025-11-01T10:30:00.123456"
}
```

### Errores Posibles

**400 Bad Request** - Email inválido o datos incorrectos
```json
{
  "detail": "Invalid email format"
}
```

**409 Conflict** - Email ya existe
```json
{
  "detail": "Email already registered"
}
```

**500 Internal Server Error** - Error del servidor
```json
{
  "detail": "Internal server error"
}
```

### Ejemplo curl

```bash
curl -X POST "http://localhost:8000/api/leads" \
  -H "Content-Type: application/json" \
  -d '{
    "nombre": "Juan Pérez",
    "email": "juan.perez@empresa.com",
    "empresa": "Retail Solutions SA",
    "sector": "Retail",
    "mensaje": "Me interesa conocer más sobre automatización"
  }'
```

### Ejemplo JavaScript (fetch)

```javascript
const createLead = async () => {
  const response = await fetch('http://localhost:8000/api/leads', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      nombre: 'Juan Pérez',
      email: 'juan.perez@empresa.com',
      empresa: 'Retail Solutions SA',
      sector: 'Retail',
      mensaje: 'Me interesa conocer más sobre automatización'
    })
  });

  if (!response.ok) {
    throw new Error('Error al crear lead');
  }

  const data = await response.json();
  return data;
};
```

### Ejemplo Python (requests)

```python
import requests

url = "http://localhost:8000/api/leads"
payload = {
    "nombre": "Juan Pérez",
    "email": "juan.perez@empresa.com",
    "empresa": "Retail Solutions SA",
    "sector": "Retail",
    "mensaje": "Me interesa conocer más sobre automatización"
}

response = requests.post(url, json=payload)
lead = response.json()
print(lead)
```

---

## 4. Obtener Lead

Obtiene la información de un lead específico por ID.

### Request

```http
GET /api/leads/{lead_id}
```

**Path Parameters:**

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `lead_id` | integer | ID del lead a obtener |

### Response

**Status Code:** `200 OK`

```json
{
  "id": 1,
  "nombre": "Juan Pérez",
  "email": "juan.perez@empresa.com",
  "empresa": "Retail Solutions SA",
  "sector": "Retail",
  "mensaje": "Me interesa conocer más sobre automatización de procesos",
  "created_at": "2025-11-01T10:30:00.123456"
}
```

### Errores Posibles

**404 Not Found** - Lead no encontrado
```json
{
  "detail": "Lead not found"
}
```

### Ejemplo curl

```bash
curl -X GET "http://localhost:8000/api/leads/1"
```

### Ejemplo JavaScript (fetch)

```javascript
const getLead = async (leadId) => {
  const response = await fetch(`http://localhost:8000/api/leads/${leadId}`);
  if (!response.ok) {
    throw new Error('Lead no encontrado');
  }
  return await response.json();
};
```

---

## 5. Crear Appointment

Crea una nueva cita de diagnóstico. Si el email no existe, crea automáticamente un lead.

### Request

```http
POST /api/appointments
Content-Type: application/json
```

**Body Parameters:**

| Campo | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| `nombre` | string | ✅ Sí | Nombre completo |
| `email` | string | ✅ Sí | Email (debe ser válido) |
| `fecha_preferida` | string (ISO 8601) | ✅ Sí | Fecha y hora preferida (formato: YYYY-MM-DDTHH:MM:SS) |
| `empresa` | string | ❌ No | Nombre de la empresa |
| `sector` | string | ❌ No | Sector de la empresa |
| `servicio_interes` | string | ❌ No | Servicio de interés |
| `mensaje` | string | ❌ No | Mensaje adicional |

**Request Body:**

```json
{
  "nombre": "María García",
  "email": "maria.garcia@empresa.com",
  "empresa": "Tech Innovations",
  "sector": "Tecnología",
  "fecha_preferida": "2025-11-15T14:00:00",
  "servicio_interes": "Desarrollo de Chatbot",
  "mensaje": "Quisiera un diagnóstico para implementar un chatbot en servicio al cliente"
}
```

### Response

**Status Code:** `201 Created`

```json
{
  "id": 1,
  "lead_id": 5,
  "fecha_preferida": "2025-11-15T14:00:00",
  "servicio_interes": "Desarrollo de Chatbot",
  "estado": "pendiente",
  "created_at": "2025-11-01T10:45:00.123456",
  "lead": {
    "id": 5,
    "nombre": "María García",
    "email": "maria.garcia@empresa.com",
    "empresa": "Tech Innovations",
    "sector": "Tecnología"
  }
}
```

### Errores Posibles

**400 Bad Request** - Datos inválidos
```json
{
  "detail": "Invalid date format. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS)"
}
```

**500 Internal Server Error** - Error del servidor
```json
{
  "detail": "Internal server error"
}
```

### Ejemplo curl

```bash
curl -X POST "http://localhost:8000/api/appointments" \
  -H "Content-Type: application/json" \
  -d '{
    "nombre": "María García",
    "email": "maria.garcia@empresa.com",
    "empresa": "Tech Innovations",
    "sector": "Tecnología",
    "fecha_preferida": "2025-11-15T14:00:00",
    "servicio_interes": "Desarrollo de Chatbot",
    "mensaje": "Quisiera un diagnóstico para chatbot"
  }'
```

### Ejemplo JavaScript (fetch)

```javascript
const createAppointment = async (appointmentData) => {
  const response = await fetch('http://localhost:8000/api/appointments', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      nombre: appointmentData.nombre,
      email: appointmentData.email,
      empresa: appointmentData.empresa,
      sector: appointmentData.sector,
      fecha_preferida: appointmentData.fecha_preferida, // ISO 8601 format
      servicio_interes: appointmentData.servicio_interes,
      mensaje: appointmentData.mensaje
    })
  });

  if (!response.ok) {
    throw new Error('Error al crear appointment');
  }

  return await response.json();
};

// Uso:
const appointment = await createAppointment({
  nombre: 'María García',
  email: 'maria.garcia@empresa.com',
  fecha_preferida: '2025-11-15T14:00:00',
  servicio_interes: 'Desarrollo de Chatbot'
});
```

---

## 6. Obtener Appointment

Obtiene la información de una cita específica por ID.

### Request

```http
GET /api/appointments/{appointment_id}
```

**Path Parameters:**

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `appointment_id` | integer | ID de la cita a obtener |

### Response

**Status Code:** `200 OK`

```json
{
  "id": 1,
  "lead_id": 5,
  "fecha_preferida": "2025-11-15T14:00:00",
  "servicio_interes": "Desarrollo de Chatbot",
  "estado": "pendiente",
  "created_at": "2025-11-01T10:45:00.123456",
  "lead": {
    "id": 5,
    "nombre": "María García",
    "email": "maria.garcia@empresa.com",
    "empresa": "Tech Innovations",
    "sector": "Tecnología"
  }
}
```

### Errores Posibles

**404 Not Found** - Cita no encontrada
```json
{
  "detail": "Appointment not found"
}
```

### Ejemplo curl

```bash
curl -X GET "http://localhost:8000/api/appointments/1"
```

---

## 7. Enviar Mensaje al Chatbot

Envía un mensaje al chatbot y recibe una respuesta inteligente.

### Request

```http
POST /api/chat
Content-Type: application/json
```

**Body Parameters:**

| Campo | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| `message` | string | ✅ Sí | Mensaje del usuario |
| `session_id` | string | ❌ No | ID de sesión (UUID). Si no se provee, se genera uno nuevo |

**Request Body:**

```json
{
  "message": "¿Qué servicios de automatización ofrecen?",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Response

**Status Code:** `200 OK`

```json
{
  "response": "¡Hola! En NEUS ofrecemos varios servicios de automatización:\n\n1. **Desarrollo de Chatbots**: Creamos asistentes virtuales inteligentes para atención al cliente, soporte técnico y más.\n\n2. **Automatización de Procesos (RPA)**: Automatizamos tareas repetitivas como procesamiento de facturas, generación de reportes, etc.\n\n3. **Modelos de IA Personalizados**: Desarrollamos soluciones de machine learning adaptadas a tus necesidades específicas.\n\n¿Te gustaría conocer más detalles sobre alguno de estos servicios?",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Errores Posibles

**400 Bad Request** - Mensaje vacío
```json
{
  "detail": "Message cannot be empty"
}
```

**500 Internal Server Error** - Error en el servicio de IA
```json
{
  "detail": "Error communicating with AI service"
}
```

### Ejemplo curl

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "¿Qué servicios de automatización ofrecen?",
    "session_id": "550e8400-e29b-41d4-a716-446655440000"
  }'
```

### Ejemplo JavaScript (fetch)

```javascript
const sendChatMessage = async (message, sessionId = null) => {
  const body = { message };
  if (sessionId) {
    body.session_id = sessionId;
  }

  const response = await fetch('http://localhost:8000/api/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body)
  });

  if (!response.ok) {
    throw new Error('Error al enviar mensaje');
  }

  return await response.json();
};

// Uso:
const chatResponse = await sendChatMessage(
  '¿Qué servicios de automatización ofrecen?',
  '550e8400-e29b-41d4-a716-446655440000'
);
console.log(chatResponse.response);
console.log('Session ID:', chatResponse.session_id);
```

### Ejemplo con React Hook

```typescript
import { useState } from 'react';

const useChat = () => {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = async (message: string) => {
    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          session_id: sessionId
        })
      });

      const data = await response.json();

      // Guardar session_id para mensajes futuros
      if (!sessionId) {
        setSessionId(data.session_id);
      }

      return data.response;
    } catch (error) {
      console.error('Error:', error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  return { sendMessage, isLoading, sessionId };
};
```

---

## 8. Obtener Historial de Chat

Obtiene el historial completo de mensajes de una sesión de chat.

### Request

```http
GET /api/chat/history/{session_id}
```

**Path Parameters:**

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `session_id` | string (UUID) | ID de la sesión de chat |

### Response

**Status Code:** `200 OK`

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "messages": [
    {
      "id": 1,
      "session_id": "550e8400-e29b-41d4-a716-446655440000",
      "message": "Hola, ¿qué servicios ofrecen?",
      "role": "user",
      "created_at": "2025-11-01T10:50:00.123456"
    },
    {
      "id": 2,
      "session_id": "550e8400-e29b-41d4-a716-446655440000",
      "message": "¡Hola! En NEUS ofrecemos...",
      "role": "assistant",
      "created_at": "2025-11-01T10:50:02.789012"
    },
    {
      "id": 3,
      "session_id": "550e8400-e29b-41d4-a716-446655440000",
      "message": "¿Cuánto cuesta el servicio de chatbot?",
      "role": "user",
      "created_at": "2025-11-01T10:51:00.456789"
    }
  ]
}
```

### Errores Posibles

**404 Not Found** - Sesión no encontrada o sin mensajes
```json
{
  "detail": "No messages found for this session"
}
```

### Ejemplo curl

```bash
curl -X GET "http://localhost:8000/api/chat/history/550e8400-e29b-41d4-a716-446655440000"
```

### Ejemplo JavaScript (fetch)

```javascript
const getChatHistory = async (sessionId) => {
  const response = await fetch(
    `http://localhost:8000/api/chat/history/${sessionId}`
  );

  if (!response.ok) {
    throw new Error('Historial no encontrado');
  }

  return await response.json();
};

// Uso:
const history = await getChatHistory('550e8400-e29b-41d4-a716-446655440000');
console.log('Mensajes:', history.messages.length);
```

---

## Códigos de Estado HTTP

| Código | Descripción | Uso |
|--------|-------------|-----|
| `200 OK` | Solicitud exitosa | GET requests exitosos |
| `201 Created` | Recurso creado exitosamente | POST requests exitosos |
| `400 Bad Request` | Datos inválidos o faltantes | Validación de datos falló |
| `404 Not Found` | Recurso no encontrado | ID inválido o recurso inexistente |
| `409 Conflict` | Conflicto (ej: email duplicado) | Violación de constraint único |
| `500 Internal Server Error` | Error del servidor | Error no manejado en backend |

---

## Estructura de Errores

Todos los errores siguen el mismo formato:

```json
{
  "detail": "Descripción del error en español"
}
```

---

## Rate Limiting

**Estado Actual:** No implementado en MVP

**Recomendado para Producción:**
- 100 requests por minuto por IP
- 1000 requests por hora por IP
- Header `X-RateLimit-Remaining` en responses

---

## CORS (Cross-Origin Resource Sharing)

El API acepta requests desde:
- `http://localhost:3000` (Create React App)
- `http://localhost:5173` (Vite)
- Configurar dominio de producción en variable `CORS_ORIGINS`

**Headers permitidos:**
- `Content-Type`
- `Authorization` (para futuras versiones)

**Métodos permitidos:**
- `GET`
- `POST`
- `PUT` (para futuras versiones)
- `DELETE` (para futuras versiones)

---

## Documentación Interactiva

El API incluye documentación interactiva generada automáticamente:

### Swagger UI
```
http://localhost:8000/docs
```
Interfaz visual para probar todos los endpoints en tiempo real.

### ReDoc
```
http://localhost:8000/redoc
```
Documentación alternativa con mejor presentación para lectura.

---

## Ejemplos de Flujos Completos

### Flujo 1: Usuario llena formulario de contacto

```javascript
// 1. Usuario llena formulario y envía
const submitContactForm = async (formData) => {
  try {
    // Crear lead
    const lead = await fetch('http://localhost:8000/api/leads', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        nombre: formData.nombre,
        email: formData.email,
        empresa: formData.empresa,
        sector: formData.sector,
        mensaje: formData.mensaje
      })
    });

    const leadData = await lead.json();
    console.log('Lead creado:', leadData.id);

    // Mostrar mensaje de éxito al usuario
    alert('¡Gracias! Te contactaremos pronto.');

  } catch (error) {
    console.error('Error:', error);
    alert('Hubo un error. Por favor intenta nuevamente.');
  }
};
```

### Flujo 2: Usuario agenda diagnóstico

```javascript
// 1. Usuario llena formulario de diagnóstico
const submitDiagnosticForm = async (formData) => {
  try {
    // Crear appointment (crea lead automáticamente si no existe)
    const appointment = await fetch('http://localhost:8000/api/appointments', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        nombre: formData.nombre,
        email: formData.email,
        empresa: formData.empresa,
        sector: formData.sector,
        fecha_preferida: formData.fecha_preferida, // "2025-11-15T14:00:00"
        servicio_interes: formData.servicio_interes,
        mensaje: formData.mensaje
      })
    });

    const appointmentData = await appointment.json();
    console.log('Cita agendada:', appointmentData.id);
    console.log('Lead asociado:', appointmentData.lead_id);

    // Mostrar confirmación
    alert(`¡Cita agendada para el ${formData.fecha_preferida}!`);

  } catch (error) {
    console.error('Error:', error);
  }
};
```

### Flujo 3: Conversación con chatbot

```javascript
// 1. Iniciar conversación
const startChat = () => {
  // Generar o recuperar session_id
  let sessionId = localStorage.getItem('chat_session_id');
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem('chat_session_id', sessionId);
  }
  return sessionId;
};

// 2. Enviar mensaje
const chatWithBot = async (userMessage) => {
  const sessionId = startChat();

  try {
    const response = await fetch('http://localhost:8000/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: userMessage,
        session_id: sessionId
      })
    });

    const data = await response.json();

    // Mostrar respuesta del bot
    console.log('Bot:', data.response);
    return data.response;

  } catch (error) {
    console.error('Error:', error);
    return 'Lo siento, hubo un error. Por favor intenta nuevamente.';
  }
};

// 3. Uso
await chatWithBot('Hola, ¿qué servicios ofrecen?');
await chatWithBot('Me interesa el servicio de chatbots');
await chatWithBot('¿Cuánto cuesta?');
```

---

## Testing del API

### Health Check Test

```bash
# Verificar que el API está corriendo
curl http://localhost:8000/api/health

# Esperado: {"status":"healthy","service":"NEUS API","version":"1.0.0"}
```

### Test Completo del Flujo

```bash
#!/bin/bash

# 1. Health check
echo "Testing health check..."
curl -X GET http://localhost:8000/api/health
echo "\n"

# 2. Crear lead
echo "Creating lead..."
curl -X POST http://localhost:8000/api/leads \
  -H "Content-Type: application/json" \
  -d '{"nombre":"Test User","email":"test@test.com","empresa":"Test Corp","sector":"Retail"}'
echo "\n"

# 3. Crear appointment
echo "Creating appointment..."
curl -X POST http://localhost:8000/api/appointments \
  -H "Content-Type: application/json" \
  -d '{"nombre":"Test User 2","email":"test2@test.com","fecha_preferida":"2025-11-15T14:00:00","servicio_interes":"Chatbot"}'
echo "\n"

# 4. Chat
echo "Testing chatbot..."
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hola"}'
echo "\n"

echo "All tests completed!"
```

---

## Changelog del API

### v1.0.0 (2025-11-01)
- Lanzamiento inicial del MVP
- 8 endpoints implementados
- Integración con Anthropic Claude para chatbot
- Documentación completa con Swagger/ReDoc

---

## Soporte y Contacto

Para reportar bugs o sugerir mejoras en el API:
1. Revisar la documentación
2. Verificar en http://localhost:8000/docs
3. Crear un issue en el repositorio
4. Contactar al equipo de desarrollo

---

**Versión:** 1.0.0
**Última Actualización:** 2025-11-01
**Autor:** NEUS Development Team
