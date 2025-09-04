import type BackendAPI from "./client";

export interface ChatSession {
  id: string;
  created_at: string;
  updated_at?: string;
  user_id: string;
  messages?: ChatMessage[];
  metadata?: Record<string, any>;
}

export interface ChatMessage {
  id?: string;
  content: string;
  role: "USER" | "ASSISTANT" | "SYSTEM" | "TOOL";
  created_at?: string;
  tool_calls?: any[];
  tool_call_id?: string;
  tokens?: {
    prompt?: number;
    completion?: number;
    total?: number;
  };
}

export interface CreateSessionRequest {
  metadata?: Record<string, any>;
}

export interface SendMessageRequest {
  message: string;
  model?: string;
  max_context_messages?: number;
}

export interface StreamChunk {
  type:
    | "text"
    | "html"
    | "error"
    | "text_chunk"
    | "tool_call"
    | "tool_response"
    | "login_needed"
    | "stream_end";
  content: string;
  // Additional fields for structured responses
  tool_id?: string;
  tool_name?: string;
  arguments?: Record<string, any>;
  result?: any;
  success?: boolean;
  message?: string;
  session_id?: string;
  agent_info?: any;
  timestamp?: string;
  summary?: any;
}

export class ChatAPI {
  private api: BackendAPI;

  constructor(api: BackendAPI) {
    this.api = api;
  }

  async createSession(request?: CreateSessionRequest): Promise<ChatSession> {
    // Generate a unique anonymous ID for this session
    const anonId =
      typeof window !== "undefined"
        ? localStorage.getItem("anon_id") ||
          Math.random().toString(36).substring(2, 15)
        : "server-anon";

    if (typeof window !== "undefined" && !localStorage.getItem("anon_id")) {
      localStorage.setItem("anon_id", anonId);
    }

    try {
      // First try with authentication if available
      const supabase = await (this.api as any).getSupabaseClient();
      const {
        data: { session },
      } = await supabase.auth.getSession();

      if (session?.access_token) {
        // User is authenticated, use normal request
        const response = await (this.api as any)._request(
          "POST",
          "/v2/chat/sessions",
          request || {},
        );
        return response;
      }
    } catch (_e) {
      // Continue with anonymous session
    }

    // Create anonymous session through proxy
    const proxyUrl = `/api/proxy/api/v2/chat/sessions`;
    const response = await fetch(proxyUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        ...request,
        metadata: { anon_id: anonId },
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Failed to create chat session: ${error}`);
    }

    return response.json();
  }

  async createSessionOld(request?: CreateSessionRequest): Promise<ChatSession> {
    const headers: HeadersInit = {
      "Content-Type": "application/json",
    };
    
    const response = await fetch(
      `/api/proxy/api/v2/chat/sessions`,
      {
        method: "POST",
        headers,
        body: JSON.stringify(request || {}),
      },
    );

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Failed to create chat session: ${error}`);
    }

    return response.json();
  }

  async getSession(
    sessionId: string,
    includeMessages = true,
  ): Promise<ChatSession> {
    const response = await (this.api as any)._get(
      `/v2/chat/sessions/${sessionId}?include_messages=${includeMessages}`,
    );
    return response;
  }

  async getSessionOld(
    sessionId: string,
    includeMessages = true,
  ): Promise<ChatSession> {
    const response = await fetch(
      `/api/proxy/api/v2/chat/sessions/${sessionId}?include_messages=${includeMessages}`,
      {
        method: "GET",
        headers,
      },
    );

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Failed to get chat session: ${error}`);
    }

    return response.json();
  }

  async listSessions(
    limit = 50,
    offset = 0,
    includeLastMessage = true,
  ): Promise<{
    sessions: ChatSession[];
    total: number;
    limit: number;
    offset: number;
  }> {
    const params = new URLSearchParams({
      limit: limit.toString(),
      offset: offset.toString(),
      include_last_message: includeLastMessage.toString(),
    });

    const response = await (this.api as any)._get(
      `/v2/chat/sessions?${params}`,
    );
    return response;
  }

  async listSessionsOld(
    limit = 50,
    offset = 0,
    includeLastMessage = true,
  ): Promise<{
    sessions: ChatSession[];
    total: number;
    limit: number;
    offset: number;
  }> {
    const params = new URLSearchParams({
      limit: limit.toString(),
      offset: offset.toString(),
      include_last_message: includeLastMessage.toString(),
    });

    const response = await fetch(
      `/api/proxy/api/v2/chat/sessions?${params}`,
      {
        method: "GET",
        headers,
      },
    );

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Failed to list chat sessions: ${error}`);
    }

    return response.json();
  }

  async deleteSession(sessionId: string): Promise<void> {
    await (this.api as any)._delete(`/v2/chat/sessions/${sessionId}`);
  }

  async deleteSessionOld(sessionId: string): Promise<void> {
    const response = await fetch(
      `/api/proxy/api/v2/chat/sessions/${sessionId}`,
      {
        method: "DELETE",
        headers,
      },
    );

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Failed to delete chat session: ${error}`);
    }
  }

  async sendMessage(
    sessionId: string,
    request: SendMessageRequest,
  ): Promise<ChatMessage> {
    const response = await (this.api as any)._request(
      "POST",
      `/v2/chat/sessions/${sessionId}/messages`,
      request,
    );
    return response;
  }

  async sendMessageOld(
    sessionId: string,
    request: SendMessageRequest,
  ): Promise<ChatMessage> {
    const response = await fetch(
      `/api/proxy/api/v2/chat/sessions/${sessionId}/messages`,
      {
        method: "POST",
        headers,
        body: JSON.stringify(request),
      },
    );

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Failed to send message: ${error}`);
    }

    return response.json();
  }

  async *streamChat(
    sessionId: string,
    message: string,
    model = "gpt-4o",
    maxContext = 50,
    onError?: (error: Error) => void,
  ): AsyncGenerator<StreamChunk, void, unknown> {
    const params = new URLSearchParams({
      message,
      model,
      max_context: maxContext.toString(),
    });

    try {
      // Use the proxy endpoint for authentication
      // The proxy will handle adding the auth token from the server session
      const proxyUrl = `/api/proxy/api/v2/chat/sessions/${sessionId}/stream?${params}`;
      
      const response = await fetch(proxyUrl, {
        method: "GET",
        // No need to set headers here - the proxy handles authentication
      });

      if (!response.ok) {
        const error = await response.text();
        throw new Error(`Failed to stream chat: ${error}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("No response body available");
      }

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");

        // Keep the last incomplete line in the buffer
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6).trim();

            if (data === "[DONE]") {
              return;
            }

            try {
              const chunk = JSON.parse(data) as StreamChunk;
              yield chunk;
            } catch (_e) {
              console.error("Failed to parse SSE data:", data, _e);
            }
          }
        }
      }
    } catch (error) {
      if (onError) {
        onError(error as Error);
      } else {
        throw error;
      }
    }
  }
}
