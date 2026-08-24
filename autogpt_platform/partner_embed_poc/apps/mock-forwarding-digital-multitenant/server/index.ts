import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { Readable } from "node:stream";
import type { ReadableStream as NodeReadableStream } from "node:stream/web";
import { fileURLToPath } from "node:url";

import cookie from "@fastify/cookie";
import fastifyStatic from "@fastify/static";
import Fastify, { type FastifyReply, type FastifyRequest } from "fastify";
import { decodeJwt } from "jose";

import {
  createPartnerAssertionIssuer,
  type PartnerIdentity,
} from "./assertion.js";
import {
  PartnerDatabase,
  type SessionView,
  type SyncMapping,
} from "./database.js";

const port = Number(process.env.PORT ?? "8788");
const host = process.env.HOST ?? "127.0.0.1";
const publicOrigin = process.env.PARTNER_ORIGIN ?? "http://localhost:8788";
const platformFrontendURL =
  process.env.AUTOGPT_FRONTEND_URL ?? "http://localhost:3000";
const platformBackendURL =
  process.env.AUTOGPT_BACKEND_URL ?? "http://localhost:8006";
const databasePath =
  process.env.DATABASE_PATH ?? resolve(process.cwd(), "data/partner.sqlite");
const assertionAudience = "autogpt-partner-exchange";
const sessionCookie = "fd_multi_session";

const store = new PartnerDatabase(databasePath);
const assertionIssuer = await createPartnerAssertionIssuer(
  publicOrigin,
  assertionAudience,
);
const app = Fastify({ logger: true });
await app.register(cookie);

app.get("/.well-known/jwks.json", async () => assertionIssuer.jwks);
app.get("/api/directory", async () => ({ users: store.directory() }));

app.post<{ Body: { userID?: string } }>(
  "/api/session",
  async (request, reply) => {
    if (typeof request.body?.userID !== "string") {
      return reply.code(400).send({ error: "A user is required" });
    }
    const sessionID = store.createSession(request.body.userID);
    if (!sessionID) return reply.code(404).send({ error: "User not found" });
    reply.setCookie(sessionCookie, sessionID, {
      httpOnly: true,
      path: "/",
      sameSite: "lax",
      secure: false,
    });
    return store.session(sessionID);
  },
);

app.get("/api/session", async (request, reply) => {
  const session = authenticatedSession(request);
  if (!session) return reply.code(401).send({ error: "Not signed in" });
  return session;
});

app.post<{ Body: { organizationID?: string } }>(
  "/api/session/organization",
  async (request, reply) => {
    const sessionID = request.cookies[sessionCookie];
    if (!sessionID) return reply.code(401).send({ error: "Not signed in" });
    if (typeof request.body?.organizationID !== "string") {
      return reply.code(400).send({ error: "An organization is required" });
    }
    if (!store.switchOrganization(sessionID, request.body.organizationID)) {
      return reply.code(403).send({ error: "Organization is not permitted" });
    }
    return store.session(sessionID);
  },
);

app.delete("/api/session", async (request, reply) => {
  const sessionID = request.cookies[sessionCookie];
  if (sessionID) store.deleteSession(sessionID);
  reply.clearCookie(sessionCookie, { path: "/" });
  return reply.code(204).send();
});

app.post("/api/autogpt/token", async (request, reply) => {
  const context = authenticationContext(request);
  if (!context) return reply.code(401).send({ error: "Not signed in" });
  const exchange = await exchangeToken(context.session, context.identity);
  return reply.code(exchange.status).send(exchange.body);
});

app.post("/api/autogpt/sync", async (request, reply) => {
  const context = authenticationContext(request);
  if (!context) return reply.code(401).send({ error: "Not signed in" });
  const exchange = await exchangeToken(context.session, context.identity);
  if (exchange.status !== 200) {
    return reply.code(exchange.status).send(exchange.body);
  }
  return { sync: exchange.mapping };
});

app.post("/api/embed/v1/sessions", async (request, reply) => {
  return proxyPlatformRequest(request, reply, "/api/embed/v1/sessions");
});

app.post<{ Params: { sessionID: string } }>(
  "/api/embed/v1/sessions/:sessionID/stream",
  async (request, reply) => {
    return proxyPlatformRequest(
      request,
      reply,
      "/api/embed/v1/sessions/" +
        encodeURIComponent(request.params.sessionID) +
        "/stream",
    );
  },
);

const staticRoot = resolve(
  dirname(fileURLToPath(import.meta.url)),
  process.env.NODE_ENV === "production" ? "../dist" : "../../dist",
);
if (existsSync(staticRoot)) {
  await app.register(fastifyStatic, { root: staticRoot });
}

await app.listen({ host, port });

function authenticatedSession(
  request: FastifyRequest,
): SessionView | undefined {
  const sessionID = request.cookies[sessionCookie];
  return sessionID ? store.session(sessionID) : undefined;
}

function authenticationContext(request: FastifyRequest):
  | {
      session: SessionView;
      identity: PartnerIdentity;
    }
  | undefined {
  const sessionID = request.cookies[sessionCookie];
  if (!sessionID) return undefined;
  const session = store.session(sessionID);
  const identity = store.identity(sessionID);
  return session && identity ? { session, identity } : undefined;
}

interface TokenBody {
  access_token: string;
  token_type: string;
  expires_in: number;
}

type ExchangeResult =
  | { status: number; body: unknown; mapping?: undefined }
  | { status: 200; body: TokenBody; mapping: SyncMapping };

async function exchangeToken(
  session: SessionView,
  identity: PartnerIdentity,
): Promise<ExchangeResult> {
  const assertion = await assertionIssuer.sign(identity);
  const response = await fetch(platformFrontendURL + "/api/embed/token", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ assertion }),
  });
  const body = (await response.json()) as unknown;
  if (!response.ok) return { status: response.status, body };
  if (!isTokenBody(body)) {
    return {
      status: 502,
      body: { error: "AutoGPT returned an invalid token response" },
    };
  }

  const claims = decodeJwt(body.access_token);
  const autoGPTUserID = claims.sub;
  const autoGPTOrganizationID = claims.organization_id;
  const autoGPTTeamID = claims.team_id;
  if (
    typeof autoGPTUserID !== "string" ||
    typeof autoGPTOrganizationID !== "string" ||
    typeof autoGPTTeamID !== "string"
  ) {
    return {
      status: 502,
      body: { error: "AutoGPT token is missing tenant mapping claims" },
    };
  }
  const mapping = store.saveMapping(
    session.user.id,
    session.activeOrganization.id,
    { autoGPTUserID, autoGPTOrganizationID, autoGPTTeamID },
  );
  return { status: 200, body, mapping };
}

function isTokenBody(value: unknown): value is TokenBody {
  if (!value || typeof value !== "object") return false;
  const token = value as Partial<TokenBody>;
  return (
    typeof token.access_token === "string" &&
    typeof token.token_type === "string" &&
    typeof token.expires_in === "number"
  );
}

async function proxyPlatformRequest(
  request: FastifyRequest,
  reply: FastifyReply,
  path: string,
) {
  const session = authenticatedSession(request);
  if (!session) return reply.code(401).send({ error: "Not signed in" });
  const authorization = request.headers.authorization;
  if (!authorization?.startsWith("Bearer ")) {
    return reply.code(401).send({ error: "Missing embed token" });
  }
  if (!tokenMatchesSession(authorization.slice(7), session)) {
    return reply
      .code(403)
      .send({ error: "Embed token does not match the active tenant" });
  }

  const response = await fetch(platformBackendURL + path, {
    method: "POST",
    headers: {
      authorization,
      "content-type": "application/json",
    },
    body: request.body === undefined ? undefined : JSON.stringify(request.body),
  });
  reply.code(response.status);
  for (const header of [
    "content-type",
    "cache-control",
    "x-vercel-ai-ui-message-stream",
  ]) {
    const value = response.headers.get(header);
    if (value) reply.header(header, value);
  }
  if (!response.body) return reply.send();
  return reply.send(
    Readable.fromWeb(response.body as unknown as NodeReadableStream),
  );
}

function tokenMatchesSession(token: string, session: SessionView): boolean {
  if (!session.sync) return false;
  try {
    const claims = decodeJwt(token);
    return (
      claims.sub === session.sync.autoGPTUserID &&
      claims.organization_id === session.sync.autoGPTOrganizationID &&
      claims.team_id === session.sync.autoGPTTeamID &&
      claims.external_account_id === session.activeOrganization.id
    );
  } catch {
    return false;
  }
}
