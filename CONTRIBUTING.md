# Guía de Contribución - NEUS

¡Gracias por tu interés en contribuir al proyecto NEUS! Esta guía te ayudará a empezar.

## Tabla de Contenidos

1. [Código de Conducta](#código-de-conducta)
2. [Cómo Contribuir](#cómo-contribuir)
3. [Reportar Bugs](#reportar-bugs)
4. [Proponer Features](#proponer-features)
5. [Workflow de Desarrollo](#workflow-de-desarrollo)
6. [Estándares de Código](#estándares-de-código)
7. [Guía de Commits](#guía-de-commits)
8. [Pull Requests](#pull-requests)

---

## Código de Conducta

Este proyecto sigue un código de conducta estricto. Al participar, te comprometes a:

- Ser respetuoso con todos los colaboradores
- Aceptar críticas constructivas
- Enfocarte en lo mejor para la comunidad
- Mostrar empatía hacia otros miembros

### Comportamiento Inaceptable

- Lenguaje ofensivo o inapropiado
- Comentarios despectivos o ataques personales
- Acoso público o privado
- Publicar información privada de otros sin permiso

**Reportar violaciones:** Contacta a los mantenedores del proyecto.

---

## Cómo Contribuir

Hay muchas formas de contribuir a NEUS:

### 1. Código
- Corregir bugs
- Implementar nuevas features
- Mejorar el rendimiento
- Refactorizar código existente

### 2. Documentación
- Mejorar README y guías
- Escribir tutoriales
- Traducir documentación
- Actualizar ejemplos

### 3. Testing
- Escribir tests unitarios
- Tests de integración
- Tests de UI
- Reportar bugs

### 4. Diseño
- Mejorar UI/UX
- Crear mockups
- Diseñar iconos o assets
- Proponer mejoras visuales

### 5. Comunidad
- Responder preguntas
- Ayudar a nuevos contribuidores
- Revisar pull requests
- Compartir el proyecto

---

## Reportar Bugs

### Antes de Reportar

1. **Busca en issues existentes** - Quizás alguien ya reportó el problema
2. **Verifica la versión** - Asegúrate de usar la última versión
3. **Reproduce el bug** - Confirma que puedes replicarlo consistentemente

### Cómo Reportar

Crea un nuevo issue con la siguiente información:

**Título:** Descripción breve y clara del bug

**Descripción:**
```markdown
## Descripción
Descripción clara del bug

## Pasos para Reproducir
1. Ir a '...'
2. Click en '...'
3. Scroll hasta '...'
4. Ver error

## Comportamiento Esperado
Qué debería pasar

## Comportamiento Actual
Qué pasa actualmente

## Screenshots
Si aplica, agregar capturas de pantalla

## Entorno
- OS: [ej: Ubuntu 22.04]
- Browser: [ej: Chrome 120]
- Versión de NEUS: [ej: 1.0.0]
- Docker version: [ej: 24.0.5]

## Logs
```
Pegar logs relevantes aquí
```

## Información Adicional
Cualquier otro contexto sobre el problema
```

**Etiquetas:** Agrega etiquetas apropiadas (`bug`, `backend`, `frontend`, etc.)

---

## Proponer Features

### Antes de Proponer

1. **Revisa el roadmap** - Verifica si ya está planificado
2. **Busca en issues** - Quizás alguien ya lo propuso
3. **Considera el alcance** - ¿Es apropiado para NEUS?

### Cómo Proponer

Crea un nuevo issue con:

**Título:** [Feature] Nombre descriptivo

**Descripción:**
```markdown
## Problema/Necesidad
¿Qué problema resuelve esta feature?

## Solución Propuesta
Descripción clara de la feature propuesta

## Alternativas Consideradas
Otras soluciones que consideraste

## Mockups/Ejemplos
Si aplica, incluir wireframes o ejemplos visuales

## Impacto
- ¿Afecta al backend? ¿Frontend? ¿Ambos?
- ¿Es breaking change?
- ¿Requiere migración de datos?

## Tareas de Implementación
- [ ] Tarea 1
- [ ] Tarea 2
- [ ] Tests
- [ ] Documentación
```

**Etiquetas:** `enhancement`, `feature-request`

---

## Workflow de Desarrollo

### 1. Fork y Clone

```bash
# Fork el repositorio en GitHub, luego:
git clone https://github.com/TU-USUARIO/neus.git
cd neus
git remote add upstream https://github.com/ORIGINAL-OWNER/neus.git
```

### 2. Crear Branch

```bash
# Actualizar main
git checkout main
git pull upstream main

# Crear branch para tu feature/fix
git checkout -b feature/nombre-descriptivo
# o
git checkout -b fix/nombre-del-bug
```

**Convención de nombres de branches:**
- `feature/` - Nueva funcionalidad
- `fix/` - Corrección de bug
- `docs/` - Cambios en documentación
- `refactor/` - Refactorización
- `test/` - Agregar o modificar tests
- `chore/` - Tareas de mantenimiento

### 3. Desarrollo Local

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
pip install -r requirements.txt
cp .env.example .env
# Configurar .env
uvicorn app.main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

**Con Docker:**
```bash
./deploy.sh
```

### 4. Hacer Cambios

- Escribe código limpio y documentado
- Sigue las guías de estilo
- Agrega tests para nuevas features
- Actualiza documentación si es necesario

### 5. Commit

```bash
git add .
git commit -m "tipo: descripción breve"
```

Ver [Guía de Commits](#guía-de-commits) para convenciones.

### 6. Push y Pull Request

```bash
git push origin feature/nombre-descriptivo
```

Luego crea un Pull Request en GitHub.

---

## Estándares de Código

### Backend (Python)

**Estilo:** PEP 8

```python
# Usar type hints
def create_lead(lead_data: LeadCreate) -> Lead:
    """
    Crea un nuevo lead en la base de datos.

    Args:
        lead_data: Datos del lead a crear

    Returns:
        Lead creado con ID asignado

    Raises:
        ValueError: Si el email ya existe
    """
    pass

# Nombres descriptivos
user_email = "test@example.com"  # ✅ Bien
e = "test@example.com"           # ❌ Mal

# Constantes en UPPER_CASE
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30

# Clases en PascalCase
class LeadService:
    pass

# Funciones y variables en snake_case
def get_user_by_email(email: str):
    pass
```

**Imports:**
```python
# Standard library
import os
from datetime import datetime

# Third party
from fastapi import FastAPI, HTTPException
from sqlalchemy.orm import Session

# Local
from app.models import Lead
from app.schemas import LeadCreate
```

**Docstrings:**
- Usar formato Google o NumPy
- Documentar todos los módulos, clases y funciones públicas

**Tests:**
- Usar pytest
- Nombrar tests con `test_` prefix
- Un test por comportamiento

### Frontend (React/TypeScript)

**Estilo:** Airbnb JavaScript Style Guide (adaptado)

```typescript
// Interfaces en PascalCase con 'I' prefix opcional
interface Lead {
  id: number;
  nombre: string;
  email: string;
}

// Componentes en PascalCase
const ContactForm: React.FC = () => {
  // Hooks al inicio
  const [email, setEmail] = useState<string>('');

  // Funciones helper
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    // ...
  };

  // Return al final
  return (
    <form onSubmit={handleSubmit}>
      {/* ... */}
    </form>
  );
};

// Props typing
interface ButtonProps {
  label: string;
  onClick: () => void;
  disabled?: boolean;
}

const Button: React.FC<ButtonProps> = ({ label, onClick, disabled = false }) => {
  return <button onClick={onClick} disabled={disabled}>{label}</button>;
};
```

**Componentes:**
- Un componente por archivo
- Usar functional components con hooks
- Nombres descriptivos
- Props typadas con TypeScript

**Styling:**
- Usar Tailwind CSS classes
- Evitar inline styles cuando sea posible
- Consistencia en espaciado y colores

**Tests:**
- Usar React Testing Library
- Tests de comportamiento, no de implementación

### SQL

```sql
-- Nombres de tablas en plural, snake_case
CREATE TABLE leads (
    id SERIAL PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Índices descriptivos
CREATE INDEX idx_leads_email ON leads(email);
```

---

## Guía de Commits

### Formato

```
tipo(scope): descripción breve

Descripción más detallada si es necesario.

- Punto relevante 1
- Punto relevante 2

Refs: #123
```

### Tipos

- `feat`: Nueva funcionalidad
- `fix`: Corrección de bug
- `docs`: Cambios en documentación
- `style`: Formateo, sin cambios de código
- `refactor`: Refactorización
- `test`: Agregar o modificar tests
- `chore`: Tareas de mantenimiento
- `perf`: Mejora de rendimiento

### Scope (opcional)

- `backend`
- `frontend`
- `db`
- `api`
- `ui`
- `chatbot`

### Ejemplos

```bash
# Feature
git commit -m "feat(chatbot): agregar soporte para mensajes multimedia"

# Fix
git commit -m "fix(api): corregir validación de email en endpoint de leads"

# Docs
git commit -m "docs: actualizar README con instrucciones de deployment"

# Refactor
git commit -m "refactor(backend): extraer lógica de chatbot a servicio separado"

# Test
git commit -m "test(frontend): agregar tests para componente ContactForm"
```

### Reglas

- Usa imperativo ("agregar" no "agregado" ni "agregando")
- Primera línea máximo 72 caracteres
- Descripción breve pero descriptiva
- Referencia issues cuando aplique (`Refs: #123`, `Closes: #45`)

---

## Pull Requests

### Antes de Crear PR

- [ ] Tu código sigue los estándares del proyecto
- [ ] Has agregado tests para nuevos cambios
- [ ] Todos los tests pasan (`npm test`, `pytest`)
- [ ] Has actualizado la documentación
- [ ] Tu branch está actualizado con `main`
- [ ] Has probado localmente

### Crear PR

**Título:** Igual que el commit principal

**Descripción:**
```markdown
## Descripción
Breve descripción de los cambios

## Tipo de Cambio
- [ ] Bug fix (non-breaking change)
- [ ] Nueva feature (non-breaking change)
- [ ] Breaking change (fix o feature que causa que funcionalidad existente no funcione)
- [ ] Documentación

## ¿Cómo se ha Testeado?
Descripción de tests realizados

## Checklist
- [ ] Mi código sigue el estilo del proyecto
- [ ] He revisado mi propio código
- [ ] He comentado áreas complejas
- [ ] He actualizado la documentación
- [ ] Mis cambios no generan nuevos warnings
- [ ] He agregado tests que prueban mi fix/feature
- [ ] Tests unitarios nuevos y existentes pasan localmente

## Screenshots (si aplica)
Agregar screenshots

## Issues Relacionados
Closes #123
Refs #456
```

### Code Review

- **Sé receptivo** a los comentarios
- **Responde** a todas las sugerencias
- **Haz cambios** solicitados prontamente
- **Agradece** el tiempo de los reviewers

### Después del Merge

- Elimina tu branch local
```bash
git checkout main
git pull upstream main
git branch -d feature/nombre-descriptivo
```

---

## Testing

### Backend Tests

```bash
cd backend
pytest
pytest --cov=app tests/  # Con coverage
```

### Frontend Tests

```bash
cd frontend
npm test
npm run test:coverage
```

### E2E Tests

```bash
# Con servicios corriendo
npm run test:e2e
```

---

## Recursos Útiles

### Documentación
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [React Docs](https://react.dev/)
- [TypeScript Docs](https://www.typescriptlang.org/docs/)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [SQLAlchemy Docs](https://docs.sqlalchemy.org/)

### Guías de Estilo
- [PEP 8](https://pep8.org/)
- [Airbnb JavaScript Style Guide](https://github.com/airbnb/javascript)
- [Conventional Commits](https://www.conventionalcommits.org/)

### Herramientas
- [Prettier](https://prettier.io/) - Formateo de código
- [ESLint](https://eslint.org/) - Linting para JavaScript/TypeScript
- [Black](https://black.readthedocs.io/) - Formateo de código Python
- [Pylint](https://pylint.org/) - Linting para Python

---

## Preguntas Frecuentes

**P: ¿Cuánto tiempo toma que revisen mi PR?**
R: Generalmente 2-5 días hábiles. Si es urgente, menciona en el PR.

**P: ¿Puedo trabajar en múltiples issues simultáneamente?**
R: Sí, pero usa branches separados para cada uno.

**P: ¿Necesito agregar tests para fixes pequeños?**
R: Sí, todos los cambios de código deberían tener tests.

**P: ¿Qué hago si mi PR tiene conflictos?**
R: Actualiza tu branch con `main` y resuelve conflictos localmente.

```bash
git checkout main
git pull upstream main
git checkout tu-branch
git merge main
# Resolver conflictos
git commit
git push
```

---

## Contacto

¿Tienes preguntas? Puedes:
- Crear un issue de tipo "Question"
- Contactar a los mantenedores
- Unirte a nuestro chat/Discord (si aplica)

---

**¡Gracias por contribuir a NEUS!**

Tu tiempo y esfuerzo hacen de este proyecto algo mejor para todos.

---

**Última Actualización:** 2025-11-01
**Versión:** 1.0.0
