import type { LeadCreate, Lead, AppointmentCreate, Appointment, ChatMessage, ChatResponse } from '../types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Generic fetch wrapper with error handling
async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  try {
    const response = await fetch(`${API_URL}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Error desconocido' }));
      throw new Error(errorData.detail || `Error ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    if (error instanceof Error) {
      throw error;
    }
    throw new Error('Error de conexión con el servidor');
  }
}

// Lead endpoints
export async function createLead(data: LeadCreate): Promise<Lead> {
  return fetchAPI<Lead>('/api/leads', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function getLead(leadId: number): Promise<Lead> {
  return fetchAPI<Lead>(`/api/leads/${leadId}`);
}

// Appointment endpoints
export async function createAppointment(data: AppointmentCreate): Promise<Appointment> {
  return fetchAPI<Appointment>('/api/appointments', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function getAppointment(appointmentId: number): Promise<Appointment> {
  return fetchAPI<Appointment>(`/api/appointments/${appointmentId}`);
}

// Chat endpoints
export async function sendChatMessage(message: string, sessionId?: string): Promise<ChatResponse> {
  const payload: ChatMessage = {
    message,
    ...(sessionId && { session_id: sessionId }),
  };

  return fetchAPI<ChatResponse>('/api/chat', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function getChatHistory(sessionId: string): Promise<any> {
  return fetchAPI<any>(`/api/chat/history/${sessionId}`);
}

// Health check
export async function checkHealth(): Promise<{ status: string }> {
  return fetchAPI<{ status: string }>('/api/health');
}
