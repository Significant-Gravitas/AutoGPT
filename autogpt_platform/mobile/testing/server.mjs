import { createHash, randomBytes, timingSafeEqual } from "node:crypto";
import { readFileSync } from "node:fs";
import { createServer as createHttpServer } from "node:http";
import { createServer as createHttpsServer } from "node:https";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const directory = new URL(".", import.meta.url);
const callbackUrl = "autogpt://auth/callback";
const html = readFileSync(new URL("fixture.html", directory), "utf8");
const script = readFileSync(new URL("fixture.js", directory), "utf8");
const authorizationScript = readFileSync(
  new URL("authorize.js", directory),
  "utf8",
);
const stylesheet = readFileSync(new URL("fixture.css", directory), "utf8");

function send(
  response,
  status,
  content,
  type = "application/json",
  headers = {},
) {
  response.writeHead(status, {
    "Content-Type": `${type}; charset=utf-8`,
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    ...headers,
  });
  response.end(typeof content === "string" ? content : JSON.stringify(content));
}

async function readBody(request) {
  let body = "";
  for await (const chunk of request) {
    body += chunk;
    if (body.length > 8192) throw new Error("Request body is too large");
  }
  return body;
}

export function createFixtureServer({
  now = Date.now,
  codeLifetimeMs = 60_000,
  tls,
} = {}) {
  const cookieName = tls
    ? "__Secure-better-auth.session_token"
    : "better-auth.session_token";
  const cacheCookieName = cookieName.replace("session_token", "session_data");
  const codes = new Map();
  const sessions = new Map();

  async function handle(request, response) {
    const origin = `${tls ? "https" : "http"}://${request.headers.host}`;
    const url = new URL(request.url, origin);
    const method = request.method;
    for (const [code, entry] of codes)
      if (entry.expiresAt <= now()) codes.delete(code);
    for (const [token, expiresAt] of sessions)
      if (expiresAt <= now()) sessions.delete(token);

    if (url.pathname === "/api/auth/mobile/start" && method === "GET") {
      const challenge = url.searchParams.get("code_challenge") ?? "";
      const state = url.searchParams.get("state") ?? "";
      if (
        !/^[A-Za-z0-9_-]{43}$/.test(challenge) ||
        !/^[A-Za-z0-9_-]{32,128}$/.test(state)
      ) {
        return send(response, 400, {
          error: "Supply a SHA-256 PKCE challenge and nonempty state",
        });
      }
      return send(response, 302, "", "text/plain", {
        Location: `/auth/mobile?${new URLSearchParams({ code_challenge: challenge, state })}`,
      });
    }

    if (url.pathname === "/auth/mobile" && method === "GET") {
      return send(
        response,
        200,
        `<!doctype html><html lang="en"><head><meta name="viewport" content="width=device-width, initial-scale=1"><title>Fixture authorization</title><link rel="stylesheet" href="/fixture.css"><script src="/authorize.js" defer></script></head><body><main><p class="eyebrow">Native integration fixture</p><h1>Connect AutoGPT</h1><p>This local fixture simulates the browser handoff. No real account is signed in.</p><button id="connect" type="button">Connect AutoGPT</button><output id="authorization-result" aria-live="polite">Ready to connect this fixture session.</output><p class="muted">Completing this returns to the native app through its registered callback.</p></main></body></html>`,
        "text/html",
      );
    }

    if (url.pathname === "/api/auth/mobile/authorize" && method === "POST") {
      if (request.headers.origin !== origin)
        return send(response, 403, {
          error: "Fixture authorization requires its configured Origin",
        });
      const body = JSON.parse(await readBody(request));
      const challenge = body?.code_challenge ?? "";
      const state = body?.state ?? "";
      if (
        !/^[A-Za-z0-9_-]{43}$/.test(challenge) ||
        !/^[A-Za-z0-9_-]{32,128}$/.test(state)
      ) {
        return send(response, 400, {
          error: "Invalid fixture authorization request",
        });
      }
      const code = randomBytes(32).toString("base64url");
      codes.set(code, { challenge, expiresAt: now() + codeLifetimeMs });
      const callback = new URL(callbackUrl);
      callback.searchParams.set("code", code);
      callback.searchParams.set("state", state);
      return send(response, 200, { url: callback.href });
    }

    if (url.pathname === "/api/auth/mobile/exchange" && method === "POST") {
      if (request.headers.origin !== origin)
        return send(response, 403, {
          error: "Fixture exchange requires its configured Origin",
        });
      let body;
      try {
        body = JSON.parse(await readBody(request));
      } catch {
        return send(response, 400, { error: "Expected a small JSON request" });
      }
      const entry = codes.get(body?.code);
      const verifier = body?.code_verifier;
      if (
        !entry ||
        typeof verifier !== "string" ||
        !/^[A-Za-z0-9._~-]{43,128}$/.test(verifier)
      ) {
        return send(response, 400, {
          error: "Invalid or expired fixture code",
        });
      }
      const supplied = createHash("sha256").update(verifier).digest();
      if (
        !timingSafeEqual(supplied, Buffer.from(entry.challenge, "base64url"))
      ) {
        return send(response, 400, {
          error: "Fixture PKCE verification failed",
        });
      }
      codes.delete(body.code);
      const token = randomBytes(24).toString("base64url");
      sessions.set(token, now() + 86_400_000);
      return send(response, 200, { success: true }, "application/json", {
        "Set-Cookie": [
          `${cookieName}=${token}; Path=/; HttpOnly; SameSite=Lax; Max-Age=86400${tls ? "; Secure" : ""}`,
          `${cacheCookieName}=fixture-cache; Path=/; HttpOnly; SameSite=Lax; Max-Age=300; Expires=${new Date(now() + 300_000).toUTCString()}${tls ? "; Secure" : ""}`,
        ],
      });
    }

    if (url.pathname === "/api/fixture/session" && method === "GET") {
      const cookie = (request.headers.cookie ?? "")
        .split(";")
        .map((part) => part.trim())
        .find((part) => part.startsWith(`${cookieName}=`));
      return send(response, 200, {
        fixture: true,
        cacheCookieReceived: (request.headers.cookie ?? "")
          .split(";")
          .some((part) => part.trim() === `${cacheCookieName}=fixture-cache`),
        authenticated: sessions.has(cookie?.slice(cookieName.length + 1)),
      });
    }

    if (url.pathname === "/api/fixture/logout" && method === "POST") {
      const cookie = (request.headers.cookie ?? "")
        .split(";")
        .map((part) => part.trim())
        .find((part) => part.startsWith(`${cookieName}=`));
      sessions.delete(cookie?.slice(cookieName.length + 1));
      return send(response, 200, { success: true }, "application/json", {
        "Set-Cookie": [cookieName, cacheCookieName].map(
          (name) =>
            `${name}=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0${tls ? "; Secure" : ""}`,
        ),
      });
    }

    if (url.pathname === "/offline") return request.socket.destroy();
    if (url.pathname === "/error")
      return send(
        response,
        503,
        "<!doctype html><title>Fixture HTTP 503</title><h1>Native integration fixture</h1><p>Intentional HTTP 503. An HTTP error page is different from a transport failure.</p><a href='/copilot'>Return to fixture</a>",
        "text/html",
      );
    if (url.pathname === "/redirect")
      return send(response, 302, "", "text/plain", {
        Location: "/copilot?redirected=1",
      });
    if (url.pathname === "/redirect/external")
      return send(response, 302, "", "text/plain", {
        Location: "https://example.com/",
      });
    if (url.pathname === "/attachment.txt")
      return send(
        response,
        200,
        "Native integration fixture attachment.\nThis file contains no account data.\n",
        "text/plain",
        { "Content-Disposition": 'attachment; filename="native-fixture.txt"' },
      );
    if (url.pathname === "/authorize.js")
      return send(response, 200, authorizationScript, "application/javascript");
    if (url.pathname === "/fixture.js")
      return send(response, 200, script, "application/javascript");
    if (url.pathname === "/fixture.css")
      return send(response, 200, stylesheet, "text/css");
    if (url.pathname === "/health")
      return send(response, 200, { fixture: true, ok: true });
    if (
      url.pathname === "/" ||
      url.pathname === "/copilot" ||
      url.pathname === "/slow"
    ) {
      if (url.pathname === "/slow")
        await new Promise((resolve) => setTimeout(resolve, 2500));
      return send(response, 200, html, "text/html");
    }
    return send(response, 404, { error: "Unknown native fixture route" });
  }

  const handler = (request, response) =>
    handle(request, response).catch(() => {
      if (!response.headersSent && !response.destroyed)
        send(response, 400, { error: "Invalid fixture request" });
    });
  return tls ? createHttpsServer(tls, handler) : createHttpServer(handler);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const { values } = parseArgs({
    options: {
      host: { type: "string", default: "127.0.0.1" },
      port: { type: "string", default: "8765" },
      "tls-cert": { type: "string" },
      "tls-key": { type: "string" },
    },
  });
  if (Boolean(values["tls-cert"]) !== Boolean(values["tls-key"]))
    throw new Error("Supply both --tls-cert and --tls-key");
  const tls = values["tls-cert"]
    ? {
        cert: readFileSync(values["tls-cert"]),
        key: readFileSync(values["tls-key"]),
      }
    : undefined;
  const port = Number(values.port);
  if (!Number.isInteger(port) || port < 1 || port > 65535)
    throw new Error("Port must be between 1 and 65535");
  const server = createFixtureServer({ tls });
  server.listen(port, values.host, () => {
    console.log(
      `Native integration fixture: ${tls ? "https" : "http"}://${values.host}:${port}/copilot`,
    );
    console.log(
      "Test-only content. No live AutoGPT account, chat, or backend is connected.",
    );
  });
}
