import type { AccessTokenSource } from "../features/auth/accessTokens";
import type { ServerConfig } from "../features/servers/serverConfig";
import { SessionRejectedError } from "./errors";
import {
  httpErrorFrom,
  openRequest,
  readBody,
  type HttpRequest,
  type OpenResponse,
} from "./http";

type Query = Record<string, string | number | boolean | null | undefined>;

export interface BackendRequest extends Omit<HttpRequest, "url"> {
  query?: Query;
}

/**
 * Authorized calls to the selected server's API origin. Paths are relative,
 * so an access token can never be attached to another host. A 401 gets one
 * retry with a freshly exchanged token; a second 401 is reported without
 * touching the stored session, which only the auth server can end.
 */
export class BackendClient {
  constructor(
    readonly server: ServerConfig,
    private readonly tokens: AccessTokenSource,
  ) {}

  async open(
    path: string,
    request: BackendRequest = {},
  ): Promise<OpenResponse> {
    const url = this.url(path, request.query);
    const first = await this.send(
      url,
      request,
      await this.tokens.getAccessToken(),
    );
    if (first.response.status !== 401) return first;
    first.release();
    const retry = await this.send(
      url,
      request,
      await this.tokens.getAccessToken({ forceRefresh: true }),
    );
    if (retry.response.status === 401) {
      retry.release();
      throw new SessionRejectedError();
    }
    return retry;
  }

  async json<T>(path: string, request: BackendRequest = {}): Promise<T> {
    const { response, release } = await this.open(path, request);
    try {
      if (!response.ok) throw await httpErrorFrom(response, request.signal);
      if (response.status === 204) return null as T;
      return (await readBody(response, request.signal)) as T;
    } finally {
      release();
    }
  }

  private send(url: string, request: BackendRequest, token: string) {
    return openRequest({
      ...request,
      url,
      headers: { ...request.headers, Authorization: `Bearer ${token}` },
    });
  }

  private url(path: string, query?: Query): string {
    if (!path.startsWith("/api/"))
      throw new Error("Backend paths must be relative API paths");
    const params = Object.entries(query ?? {})
      .filter(
        (entry): entry is [string, string | number | boolean] =>
          entry[1] != null,
      )
      .map(
        ([key, value]) =>
          `${encodeURIComponent(key)}=${encodeURIComponent(String(value))}`,
      );
    return `${this.server.apiUrl}${path}${params.length ? `?${params.join("&")}` : ""}`;
  }
}
