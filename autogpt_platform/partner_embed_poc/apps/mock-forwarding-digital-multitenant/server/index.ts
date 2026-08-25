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
import {
  createDemoAccessGate,
  createDemoAccessRateLimiter,
  DEMO_ACCESS_TTL_SECONDS,
} from "./demo-access.js";

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
const demoAccessCookie = "partner_demo_access";
const publicDemoMode = process.env.DEMO_PUBLIC_MODE === "true";
const demoAccessGate = createDemoAccessGate(process.env.DEMO_ACCESS_CODE, {
  required: publicDemoMode,
});
const demoAccessRateLimiter = createDemoAccessRateLimiter();
const secureDemoAccessCookie = process.env.DEMO_ACCESS_COOKIE_SECURE === "true";
if (publicDemoMode && !secureDemoAccessCookie) {
  throw new Error("DEMO_ACCESS_COOKIE_SECURE must be true in public demo mode");
}

const store = new PartnerDatabase(databasePath);
const assertionIssuer = await createPartnerAssertionIssuer(
  publicOrigin,
  assertionAudience,
);
const app = Fastify({ logger: true, trustProxy: publicDemoMode ? 1 : false });
await app.register(cookie);
app.addHook("preHandler", async (request, reply) => {
  const requestPath = request.url.split("?", 1)[0];
  if (
    !demoAccessGate.enabled ||
    !requestPath.startsWith("/api/") ||
    requestPath === "/api/demo-access"
  ) {
    return;
  }
  if (demoAccessGate.acceptsCookie(request.cookies[demoAccessCookie])) return;
  return reply.code(401).send({ error: "Demo access required" });
});

app.get("/api/demo-access", async (request, reply) => {
  reply.header("cache-control", "no-store");
  return {
    required: demoAccessGate.enabled,
    authorized:
      !demoAccessGate.enabled ||
      demoAccessGate.acceptsCookie(request.cookies[demoAccessCookie]),
  };
});

app.post<{ Body: { code?: string } }>(
  "/api/demo-access",
  async (request, reply) => {
    reply.header("cache-control", "no-store");
    if (!demoAccessGate.enabled) return reply.code(204).send();
    const attempt = demoAccessRateLimiter.consume(request.ip);
    if (!attempt.allowed) {
      reply.header("retry-after", String(attempt.retryAfterSeconds));
      return reply
        .code(429)
        .send({ error: "Too many demo access attempts. Try again later." });
    }
    if (
      typeof request.body?.code !== "string" ||
      !demoAccessGate.acceptsCode(request.body.code)
    ) {
      return reply.code(401).send({ error: "Invalid demo access code" });
    }
    demoAccessRateLimiter.reset(request.ip);
    reply.setCookie(demoAccessCookie, demoAccessGate.cookieValue(), {
      httpOnly: true,
      maxAge: DEMO_ACCESS_TTL_SECONDS,
      path: "/",
      sameSite: "strict",
      secure: secureDemoAccessCookie,
    });
    return reply.code(204).send();
  },
);

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
      sameSite: "strict",
      secure: secureDemoAccessCookie,
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

app.get("/api/embed/v1/sessions", async (request, reply) => {
  return proxyPlatformRequest(request, reply, "/api/embed/v1/sessions");
});

app.get<{ Params: { sessionID: string } }>(
  "/api/embed/v1/sessions/:sessionID",
  async (request, reply) => {
    return proxyPlatformRequest(
      request,
      reply,
      "/api/embed/v1/sessions/" + encodeURIComponent(request.params.sessionID),
    );
  },
);

app.get<{ Params: { sessionID: string } }>(
  "/api/embed/v1/sessions/:sessionID/artifacts",
  async (request, reply) => {
    return proxyPlatformRequest(
      request,
      reply,
      "/api/embed/v1/sessions/" +
        encodeURIComponent(request.params.sessionID) +
        "/artifacts",
    );
  },
);

app.get<{ Params: { sessionID: string; fileID: string } }>(
  "/api/embed/v1/sessions/:sessionID/artifacts/:fileID/download",
  async (request, reply) => {
    return proxyPlatformRequest(
      request,
      reply,
      "/api/embed/v1/sessions/" +
        encodeURIComponent(request.params.sessionID) +
        "/artifacts/" +
        encodeURIComponent(request.params.fileID) +
        "/download",
    );
  },
);

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

const staticRoot = process.env.STATIC_ROOT
  ? resolve(process.env.STATIC_ROOT)
  : resolve(
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

  const queryIndex = request.url.indexOf("?");
  const query = queryIndex >= 0 ? request.url.slice(queryIndex) : "";
  const response = await fetch(platformBackendURL + path + query, {
    method: request.method,
    headers: {
      authorization,
      "content-type": "application/json",
    },
    body:
      request.method === "GET" || request.body === undefined
        ? undefined
        : JSON.stringify(request.body),
  });
  reply.code(response.status);
  for (const header of [
    "content-type",
    "content-disposition",
    "content-length",
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
