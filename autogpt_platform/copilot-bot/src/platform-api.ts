/**
 * Client for the AutoGPT Platform Linking & Chat APIs.
 *
 * Handles:
 * - Resolving platform users → AutoGPT accounts
 * - Creating link tokens for unlinked users
 * - Checking link token status
 * - Creating chat sessions and streaming messages
 */

export interface ResolveResult {
  linked: boolean;
  user_id?: string;
  platform_username?: string;
}

export interface LinkTokenResult {
  token: string;
  expires_at: string;
  link_url: string;
}

export interface LinkTokenStatus {
  status: "pending" | "linked" | "expired";
  user_id?: string;
}

export class PlatformAPI {
  constructor(private baseUrl: string) {}

  /**
   * Check if a platform user is linked to an AutoGPT account.
   */
  async resolve(
    platform: string,
    platformUserId: string
  ): Promise<ResolveResult> {
    const res = await fetch(`${this.baseUrl}/api/platform-linking/resolve`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...this.botHeaders(),
      },
      body: JSON.stringify({
        platform: platform.toUpperCase(),
        platform_user_id: platformUserId,
      }),
    });

    if (!res.ok) {
      throw new Error(
        `Platform resolve failed: ${res.status} ${await res.text()}`
      );
    }

    return res.json();
  }

  /**
   * Create a link token for an unlinked platform user.
   */
  async createLinkToken(params: {
    platform: string;
    platformUserId: string;
    platformUsername?: string;
    channelId?: string;
  }): Promise<LinkTokenResult> {
    const res = await fetch(`${this.baseUrl}/api/platform-linking/tokens`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...this.botHeaders(),
      },
      body: JSON.stringify({
        platform: params.platform.toUpperCase(),
        platform_user_id: params.platformUserId,
        platform_username: params.platformUsername,
        channel_id: params.channelId,
      }),
    });

    if (!res.ok) {
      throw new Error(
        `Create link token failed: ${res.status} ${await res.text()}`
      );
    }

    return res.json();
  }

  /**
   * Check if a link token has been consumed.
   */
  async getLinkTokenStatus(token: string): Promise<LinkTokenStatus> {
    const res = await fetch(
      `${this.baseUrl}/api/platform-linking/tokens/${token}/status`,
      { headers: this.botHeaders() }
    );

    if (!res.ok) {
      throw new Error(
        `Link token status failed: ${res.status} ${await res.text()}`
      );
    }

    return res.json();
  }

  /**
   * Create a new CoPilot chat session for a linked user.
   * Uses the bot chat proxy (no user JWT needed).
   */
  async createChatSession(userId: string): Promise<string> {
    const res = await fetch(
      `${this.baseUrl}/api/platform-linking/chat/session`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...this.botHeaders(),
        },
        body: JSON.stringify({
          user_id: userId,
          message: "session_init",
        }),
      }
    );

    if (!res.ok) {
      throw new Error(
        `Create chat session failed: ${res.status} ${await res.text()}`
      );
    }

    const data = await res.json();
    return data.session_id;
  }

  /**
   * Stream a chat message to CoPilot on behalf of a linked user.
   * Uses the bot chat proxy — authenticated via bot API key.
   * Yields text chunks from the SSE stream.
   */
  async *streamChat(
    userId: string,
    message: string,
    sessionId?: string
  ): AsyncGenerator<string> {
    const res = await fetch(
      `${this.baseUrl}/api/platform-linking/chat/stream`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
          ...this.botHeaders(),
        },
        body: JSON.stringify({
          user_id: userId,
          message,
          session_id: sessionId,
        }),
      }
    );

    if (!res.ok) {
      throw new Error(
        `Stream chat failed: ${res.status} ${await res.text()}`
      );
    }

    if (!res.body) {
      throw new Error("No response body for SSE stream");
    }

    // Parse SSE stream
    const decoder = new TextDecoder();
    const reader = res.body.getReader();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const data = line.slice(6).trim();
          if (data === "[DONE]") return;

          try {
            const parsed = JSON.parse(data);
            // Backend sends: text_delta (streaming chunks), text_start/text_end,
            // start/finish (lifecycle), step_start/step_finish, error, etc.
            if (parsed.type === "text_delta" && parsed.delta) {
              yield parsed.delta;
            } else if (parsed.type === "error" && parsed.content) {
              yield `Error: ${parsed.content}`;
            }
            // Ignore start/finish/step lifecycle events — they carry no text
          } catch {
            // Non-JSON data line — skip
          }
        }
      }
    }
  }

  private botHeaders(): Record<string, string> {
    const key = process.env.PLATFORM_BOT_API_KEY;
    if (key) {
      return { "X-Bot-API-Key": key };
    }
    return {};
  }
}
