import { randomUUID } from "node:crypto";

import Fastify, { type FastifyReply, type FastifyRequest } from "fastify";

import {
  bearerToken,
  verifyAccessToken,
  type PartnerMCPClaims,
} from "./auth.js";
import { report, tenantExists, type ReportName } from "./data.js";

interface JSONRPCRequest {
  jsonrpc?: string;
  id?: string | number | null;
  method?: string;
  params?: unknown;
}

interface MCPServerOptions {
  sharedSecret: string;
}

const TOOLS = [
  {
    name: "get_operations_summary",
    description:
      "Return the current tenant's operational and financial summary.",
    inputSchema: {
      type: "object",
      properties: {},
      additionalProperties: false,
    },
  },
  {
    name: "list_arrivals",
    description: "List the current tenant's upcoming freight arrivals.",
    inputSchema: {
      type: "object",
      properties: {},
      additionalProperties: false,
    },
  },
  {
    name: "list_exceptions",
    description: "List the current tenant's open freight exceptions.",
    inputSchema: {
      type: "object",
      properties: {},
      additionalProperties: false,
    },
  },
];

const TOOL_CAPABILITIES: Record<ReportName, string> = {
  get_operations_summary: "reports.read",
  list_arrivals: "jobs.read",
  list_exceptions: "jobs.read",
};

export function buildApp(options: MCPServerOptions) {
  if (options.sharedSecret.length < 16) {
    throw new Error("MCP shared secret must contain at least 16 characters");
  }
  const sessions = new Map<string, string>();
  const app = Fastify({ logger: false });

  app.get("/health", async () => ({ status: "ok" }));

  app.post<{ Body: JSONRPCRequest }>("/mcp", async (request, reply) => {
    const claims = authenticate(request, reply, options.sharedSecret);
    if (!claims) return;
    const body = request.body;
    if (body?.jsonrpc !== "2.0" || typeof body.method !== "string") {
      return rpcError(reply, body?.id ?? null, -32600, "Invalid Request");
    }
    if (!tenantExists(claims.external_account_id)) {
      return reply.code(403).send({ error: "Tenant is not configured" });
    }

    if (body.method === "initialize") {
      const sessionID = randomUUID();
      sessions.set(sessionID, sessionBinding(claims));
      return reply.header("Mcp-Session-Id", sessionID).send(
        rpcResult(body.id ?? null, {
          protocolVersion: "2025-03-26",
          capabilities: { tools: { listChanged: false } },
          serverInfo: {
            name: "logistics-partner-tenant-mcp",
            version: "0.1.0",
          },
        }),
      );
    }

    const sessionID = sessionHeader(request);
    if (!sessionID || !sessions.has(sessionID)) {
      return reply.code(400).send({ error: "Valid MCP session required" });
    }
    if (sessions.get(sessionID) !== sessionBinding(claims)) {
      return reply.code(403).send({ error: "MCP session tenant mismatch" });
    }

    if (body.method === "notifications/initialized") {
      return reply.code(202).send();
    }
    if (body.method === "tools/list") {
      return reply.send(
        rpcResult(body.id ?? null, {
          tools: TOOLS.filter((tool) =>
            claims.capabilities.includes(
              TOOL_CAPABILITIES[tool.name as ReportName],
            ),
          ),
        }),
      );
    }
    if (body.method === "tools/call") {
      return callTool(reply, body, claims);
    }
    return rpcError(reply, body.id ?? null, -32601, "Method not found");
  });

  app.delete("/mcp", async (request, reply) => {
    const claims = authenticate(request, reply, options.sharedSecret);
    if (!claims) return;
    const sessionID = sessionHeader(request);
    if (!sessionID || sessions.get(sessionID) !== sessionBinding(claims)) {
      return reply.code(404).send({ error: "MCP session not found" });
    }
    sessions.delete(sessionID);
    return reply.code(204).send();
  });

  return app;
}

function callTool(
  reply: FastifyReply,
  request: JSONRPCRequest,
  claims: PartnerMCPClaims,
) {
  if (!request.params || typeof request.params !== "object") {
    return rpcError(reply, request.id ?? null, -32602, "Invalid params");
  }
  const params = request.params as {
    name?: unknown;
    arguments?: unknown;
  };
  if (!isReportName(params.name) || !isEmptyObject(params.arguments)) {
    return rpcError(reply, request.id ?? null, -32602, "Invalid params");
  }
  if (!claims.capabilities.includes(TOOL_CAPABILITIES[params.name])) {
    return rpcError(reply, request.id ?? null, -32001, "Capability denied");
  }
  const result = report(claims.external_account_id, params.name);
  return reply.send(
    rpcResult(request.id ?? null, {
      content: [{ type: "text", text: JSON.stringify(result) }],
      structuredContent: result,
      isError: false,
    }),
  );
}

function authenticate(
  request: FastifyRequest,
  reply: FastifyReply,
  secret: string,
) {
  const token = bearerToken(request.headers.authorization);
  const claims = token ? verifyAccessToken(token, secret) : undefined;
  if (!claims) {
    reply.code(401).send({ error: "Invalid partner MCP bearer token" });
    return undefined;
  }
  return claims;
}

function sessionHeader(request: FastifyRequest): string | undefined {
  const value = request.headers["mcp-session-id"];
  return typeof value === "string" ? value : value?.[0];
}

function sessionBinding(claims: PartnerMCPClaims): string {
  return JSON.stringify([
    claims.external_account_id,
    [...new Set(claims.capabilities)].sort(),
  ]);
}

function isReportName(value: unknown): value is ReportName {
  return (
    value === "get_operations_summary" ||
    value === "list_arrivals" ||
    value === "list_exceptions"
  );
}

function isEmptyObject(value: unknown): value is Record<string, never> {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    Object.keys(value).length === 0
  );
}

function rpcResult(id: JSONRPCRequest["id"], result: unknown) {
  return { jsonrpc: "2.0", id, result };
}

function rpcError(
  reply: FastifyReply,
  id: JSONRPCRequest["id"],
  code: number,
  message: string,
) {
  return reply.send({ jsonrpc: "2.0", id, error: { code, message } });
}
