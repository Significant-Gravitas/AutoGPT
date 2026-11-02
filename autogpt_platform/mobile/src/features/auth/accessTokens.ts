import { SessionEndedError } from "../../api/errors";
import type { ServerConfig } from "../servers/serverConfig";
import { exchangeAccessToken } from "./authApi";

export interface AccessTokenSource {
  getAccessToken(options?: { forceRefresh?: boolean }): Promise<string>;
}

interface Hooks {
  /** The auth server reported the session gone; the provider signs out locally. */
  onSessionEnded: () => void;
  onSessionRenewed: (sessionToken: string) => void;
}

const REFRESH_MARGIN_MS = 60_000;

/**
 * Short-lived API access tokens for one signed-in account. Tokens live only
 * in memory; concurrent callers share a single exchange.
 */
export class SessionAccessTokens implements AccessTokenSource {
  private cached: { token: string; expiresAt: number } | null = null;
  private inflight: Promise<string> | null = null;
  private disposed = false;

  constructor(
    private readonly server: ServerConfig,
    private sessionToken: string,
    private readonly hooks: Hooks,
  ) {}

  async getAccessToken({ forceRefresh = false } = {}): Promise<string> {
    if (this.disposed) throw new SessionEndedError();
    const cached = this.cached;
    if (
      !forceRefresh &&
      cached &&
      cached.expiresAt - REFRESH_MARGIN_MS > Date.now()
    ) {
      return cached.token;
    }
    if (!this.inflight) {
      this.inflight = this.refresh().finally(() => {
        this.inflight = null;
      });
    }
    return this.inflight;
  }

  /** Reuses the token sign-in already obtained while verifying the session. */
  seed(token: string, expiresAt: number) {
    if (!this.disposed) this.cached = { token, expiresAt };
  }

  dispose() {
    this.disposed = true;
    this.cached = null;
  }

  private async refresh(): Promise<string> {
    try {
      const result = await exchangeAccessToken(this.server, this.sessionToken);
      if (this.disposed) throw new SessionEndedError();
      this.cached = { token: result.token, expiresAt: result.expiresAt };
      if (result.renewedSessionToken) {
        this.sessionToken = result.renewedSessionToken;
        this.hooks.onSessionRenewed(result.renewedSessionToken);
      }
      return result.token;
    } catch (error) {
      if (error instanceof SessionEndedError && !this.disposed) {
        this.cached = null;
        this.hooks.onSessionEnded();
      }
      throw error;
    }
  }
}
