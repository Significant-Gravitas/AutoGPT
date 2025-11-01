# NEUS Frontend

Frontend moderno para la plataforma de servicios de IA empresarial NEUS, construido con React, TypeScript y Tailwind CSS.

## Tecnologías Utilizadas

- **React 18** - Librería de UI
- **TypeScript** - Tipado estático
- **Vite** - Build tool y dev server
- **Tailwind CSS** - Framework de estilos utility-first
- **Lucide React** - Iconos modernos
- **UUID** - Generación de IDs únicos para sesiones de chat

## Estructura del Proyecto

```
frontend/
├── public/
│   └── vite.svg                    # Logo/favicon
├── src/
│   ├── components/
│   │   ├── Chatbot/
│   │   │   ├── ChatWidget.tsx      # Widget principal de chat
│   │   │   ├── ChatMessage.tsx     # Componente de mensaje individual
│   │   │   └── ChatInput.tsx       # Input de chat
│   │   ├── Hero.tsx                # Hero section
│   │   ├── Services.tsx            # Sección de servicios
│   │   ├── Sectors.tsx             # Sección de sectores
│   │   ├── WhyNeus.tsx             # Razones para elegir NEUS
│   │   ├── ContactForm.tsx         # Formulario de contacto
│   │   ├── DiagnosticForm.tsx      # Formulario de diagnóstico
│   │   ├── Footer.tsx              # Footer
│   │   └── Navbar.tsx              # Barra de navegación
│   ├── services/
│   │   └── api.ts                  # Funciones para API calls
│   ├── types/
│   │   └── index.ts                # Tipos TypeScript
│   ├── App.tsx                     # Componente principal
│   ├── main.tsx                    # Punto de entrada
│   └── index.css                   # Estilos globales + Tailwind
├── .env.example                    # Template de variables de entorno
├── package.json                    # Dependencias
├── tsconfig.json                   # Configuración TypeScript
├── vite.config.ts                  # Configuración Vite
├── tailwind.config.js              # Configuración Tailwind
├── postcss.config.js               # Configuración PostCSS
├── Dockerfile                      # Containerización
└── README.md                       # Este archivo
```

## Características Principales

### 1. Landing Page Moderna y Responsive
- Hero section impactante con gradientes
- Secciones claramente definidas (Servicios, Sectores, Por qué NEUS)
- Diseño mobile-first completamente responsive
- Animaciones y transiciones suaves

### 2. Formularios de Contacto y Diagnóstico
- **Formulario de Contacto**: Captura leads básicos
- **Formulario de Diagnóstico**: Agenda citas con más detalle
- Validación en tiempo real
- Estados de carga y mensajes de éxito/error
- Conectados al backend mediante API REST

### 3. Chatbot Inteligente
- Widget flotante en la esquina inferior derecha
- Ventana de chat expandible
- Mantiene sesiones usando localStorage
- Indicador visual de "escribiendo..."
- Scroll automático a nuevos mensajes
- Integración con Claude API mediante el backend

### 4. Integración con Backend
- Endpoints REST consumidos:
  - `POST /api/leads` - Crear lead de contacto
  - `POST /api/appointments` - Agendar diagnóstico
  - `POST /api/chat` - Enviar mensaje al chatbot
- Manejo de errores robusto
- Variables de entorno para configuración

## Configuración

### Variables de Entorno

Copia el archivo `.env.example` a `.env` y configura:

```bash
cp .env.example .env
```

```env
VITE_API_URL=http://localhost:8000
```

## Instalación y Desarrollo

### Requisitos Previos
- Node.js 18 o superior
- npm o yarn

### Instalación

```bash
# Instalar dependencias
npm install
```

### Desarrollo

```bash
# Ejecutar en modo desarrollo (http://localhost:5173)
npm run dev
```

El servidor de desarrollo se iniciará en `http://localhost:5173` con hot reload activado.

### Build para Producción

```bash
# Crear build de producción
npm run build

# Preview del build
npm run preview
```

Los archivos compilados se generarán en el directorio `dist/`.

## Docker

### Build de la imagen

```bash
docker build -t neus-frontend .
```

### Ejecutar contenedor

```bash
docker run -p 80:80 neus-frontend
```

La aplicación estará disponible en `http://localhost`.

### Build multi-stage

El Dockerfile utiliza un build multi-stage:
1. **Stage 1 (build)**: Compila la aplicación con Node.js
2. **Stage 2 (production)**: Sirve los archivos estáticos con nginx

Esto resulta en una imagen de producción muy ligera (~25MB).

## Componentes Principales

### Hero.tsx
Sección principal con:
- Título impactante con gradientes
- Propuesta de valor clara
- CTAs para diagnóstico gratuito
- Estadísticas destacadas

### Services.tsx
Muestra los 4 pilares de servicio:
- Capacitación en IA
- Consultoría Estratégica
- Desarrollo y Automatización
- Infraestructura y Seguridad

### Sectors.tsx
Grid con los 8 sectores especializados con iconos.

### WhyNeus.tsx
Lista las 5 razones para elegir NEUS con iconos y descripciones detalladas.

### ContactForm.tsx
Formulario de contacto conectado a `POST /api/leads`:
- Campos: nombre*, email*, empresa, sector, mensaje
- Validación HTML5
- Estados de carga/éxito/error

### DiagnosticForm.tsx
Modal para agendar diagnósticos, conectado a `POST /api/appointments`:
- Campos: nombre*, email*, empresa, sector, fecha_preferida*, servicio_interes, mensaje
- Date picker con validación de fechas futuras
- Modal responsive

### ChatWidget
Widget de chat flotante:
- Sesión persistente en localStorage
- Integración con API de chat
- UX pulida con indicadores de carga
- Auto-scroll a nuevos mensajes

## Paleta de Colores

```css
/* Definidos en tailwind.config.js */
primary-600: #0066FF    /* Azul tecnológico */
secondary-600: #6B21A8  /* Morado/violeta */

/* Gradientes usados */
bg-gradient-to-r from-primary-600 to-secondary-600
```

## Estilos y Temas

- Tailwind CSS con configuración personalizada
- Gradientes de azul a morado para elementos tecnológicos
- Sombras y efectos hover para mejor UX
- Diseño responsive con breakpoints:
  - sm: 640px
  - md: 768px
  - lg: 1024px
  - xl: 1280px

## API Integration

El archivo `src/services/api.ts` contiene todas las funciones para interactuar con el backend:

```typescript
// Crear lead
await createLead({ nombre, email, empresa?, sector?, mensaje? });

// Agendar diagnóstico
await createAppointment({ nombre, email, fecha_preferida, ... });

// Enviar mensaje al chatbot
await sendChatMessage(message, sessionId?);
```

## Deployment

### Opción 1: Build estático + servidor web

```bash
npm run build
# Copiar dist/ a tu servidor web
```

### Opción 2: Docker

```bash
docker build -t neus-frontend .
docker run -p 80:80 neus-frontend
```

### Opción 3: Vercel/Netlify

El proyecto está listo para deployarse en plataformas como Vercel o Netlify:
- Build command: `npm run build`
- Output directory: `dist`
- Environment variable: `VITE_API_URL`

## Notas para el Siguiente Agente (Infraestructura)

El frontend está completamente funcional y listo para deployment. Consideraciones:

1. **Variables de entorno**: Configurar `VITE_API_URL` apuntando al backend en producción
2. **CORS**: El backend ya está configurado para aceptar peticiones del frontend
3. **Nginx**: Si usas nginx, considera añadir configuración para SPA (redirect a index.html)
4. **Docker Compose**: Puedes orquestar frontend + backend + database
5. **CDN**: Los assets estáticos pueden servirse desde CDN para mejor rendimiento
6. **HTTPS**: Configurar certificados SSL para producción

## Licencia

© 2025 NEUS. Todos los derechos reservados.
