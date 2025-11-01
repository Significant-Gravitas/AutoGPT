// Types matching backend schemas

export interface LeadCreate {
  nombre: string;
  email: string;
  empresa?: string;
  sector?: string;
  mensaje?: string;
}

export interface Lead extends LeadCreate {
  id: number;
  created_at: string;
}

export interface AppointmentCreate {
  nombre: string;
  email: string;
  empresa?: string;
  sector?: string;
  fecha_preferida: string; // ISO date string
  servicio_interes?: string;
  mensaje?: string;
}

export interface Appointment {
  id: number;
  lead_id: number;
  fecha_preferida: string;
  servicio_interes: string | null;
  estado: string;
  created_at: string;
}

export interface ChatMessage {
  message: string;
  session_id?: string;
}

export interface ChatResponse {
  response: string;
  session_id: string;
}

// Sectores disponibles
export const SECTORES = [
  'Retail',
  'Salud',
  'Supply Chain',
  'Administración Pública',
  'Legal',
  'Onboarding',
  'Back-Office',
  'Formación'
] as const;

export type Sector = typeof SECTORES[number];

// Servicios disponibles
export const SERVICIOS = [
  'Capacitación en IA',
  'Consultoría Estratégica',
  'Desarrollo y Automatización',
  'Infraestructura y Seguridad'
] as const;

export type Servicio = typeof SERVICIOS[number];
