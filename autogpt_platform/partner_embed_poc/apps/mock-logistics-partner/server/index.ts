import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { Readable } from "node:stream";
import type { ReadableStream as NodeReadableStream } from "node:stream/web";
import { fileURLToPath } from "node:url";

import cookie from "@fastify/cookie";
import fastifyStatic from "@fastify/static";
import Fastify, { type FastifyReply, type FastifyRequest } from "fastify";

import { createPartnerAssertionIssuer, type PartnerUser } from "./assertion.js";

const port = Number(process.env.PORT ?? "8787");
const host = process.env.HOST ?? "127.0.0.1";
const publicOrigin = process.env.PARTNER_ORIGIN ?? `http://localhost:${port}`;
const platformFrontendURL =
  process.env.AUTOGPT_FRONTEND_URL ?? "http://localhost:3000";
const platformBackendURL =
  process.env.AUTOGPT_BACKEND_URL ?? "http://localhost:8006";
const assertionAudience = "autogpt-partner-exchange";
const sessionCookie = "fd_poc_session";
const sessions = new Map<string, PartnerUser>();
const demoUser: PartnerUser = {
  subject: "fd-user-1042",
  accountID: "fd-account-77",
  email: "alex@northstarfreight.com",
  name: "Alex Morgan",
  accountName: "Northstar Freight",
  roles: ["operator", "manager"],
  capabilities: [
    "jobs.read",
    "reports.read",
    "documents.read",
    "documents.write",
    "agents.create",
    "agents.run",
    "agents.schedule",
    "autogpt:block:c0a8e994-ebf1-4a9c-a4d8-89d09c86741b",
    "autogpt:block:363ae599-353e-4804-937e-b2ee3cef3da4",
    "autogpt:block:b1ab9b19-67a6-406d-abf5-2dba76d00c79",
  ],
};

const assertionIssuer = await createPartnerAssertionIssuer(
  publicOrigin,
  assertionAudience,
);
const app = Fastify({ logger: true });
await app.register(cookie);

app.get("/.well-known/jwks.json", async () => assertionIssuer.jwks);

app.post("/api/session", async (_request, reply) => {
  const sessionID = randomUUID();
  sessions.set(sessionID, demoUser);
  reply.setCookie(sessionCookie, sessionID, {
    httpOnly: true,
    path: "/",
    sameSite: "lax",
    secure: false,
  });
  return publicUser(demoUser);
});

app.get("/api/session", async (request, reply) => {
  const user = authenticatedUser(request);
  if (!user) return reply.code(401).send({ error: "Not signed in" });
  return publicUser(user);
});

app.delete("/api/session", async (request, reply) => {
  const sessionID = request.cookies[sessionCookie];
  if (sessionID) sessions.delete(sessionID);
  reply.clearCookie(sessionCookie, { path: "/" });
  return reply.code(204).send();
});

app.post("/api/autogpt/token", async (request, reply) => {
  const user = authenticatedUser(request);
  if (!user) return reply.code(401).send({ error: "Not signed in" });

  const assertion = await assertionIssuer.sign(user);
  const response = await fetch(`${platformFrontendURL}/api/embed/token`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ assertion }),
  });
  const body = await response.text();
  reply
    .code(response.status)
    .header(
      "content-type",
      response.headers.get("content-type") ?? "application/json",
    );
  return reply.send(body);
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
      `/api/embed/v1/sessions/${encodeURIComponent(request.params.sessionID)}`,
    );
  },
);

app.get<{ Params: { sessionID: string } }>(
  "/api/embed/v1/sessions/:sessionID/artifacts",
  async (request, reply) => {
    return proxyPlatformRequest(
      request,
      reply,
      `/api/embed/v1/sessions/${encodeURIComponent(request.params.sessionID)}/artifacts`,
    );
  },
);

app.get<{ Params: { sessionID: string; fileID: string } }>(
  "/api/embed/v1/sessions/:sessionID/artifacts/:fileID/download",
  async (request, reply) => {
    return proxyPlatformRequest(
      request,
      reply,
      `/api/embed/v1/sessions/${encodeURIComponent(request.params.sessionID)}/artifacts/${encodeURIComponent(request.params.fileID)}/download`,
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
      `/api/embed/v1/sessions/${encodeURIComponent(request.params.sessionID)}/stream`,
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

function authenticatedUser(request: FastifyRequest): PartnerUser | undefined {
  const sessionID = request.cookies[sessionCookie];
  return sessionID ? sessions.get(sessionID) : undefined;
}

function publicUser(user: PartnerUser) {
  return {
    name: user.name,
    email: user.email,
    accountName: user.accountName,
    roles: user.roles,
  };
}

async function proxyPlatformRequest(
  request: FastifyRequest,
  reply: FastifyReply,
  path: string,
) {
  if (!authenticatedUser(request)) {
    return reply.code(401).send({ error: "Not signed in" });
  }
  const authorization = request.headers.authorization;
  if (!authorization) {
    return reply.code(401).send({ error: "Missing embed token" });
  }

  const queryIndex = request.url.indexOf("?");
  const query = queryIndex >= 0 ? request.url.slice(queryIndex) : "";
  const response = await fetch(`${platformBackendURL}${path}${query}`, {
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
