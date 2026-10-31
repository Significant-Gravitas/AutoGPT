// Targets for the AutoGPT Platform load tests, picked by K6_ENVIRONMENT
// (default PREVIEW, which also needs PR_NUMBER).
//
// - BACKEND_URL: the backend's origin; /health lives here.
// - API_BASE_URL: the backend's REST root (BACKEND_URL + /api).
// - FRONTEND_URL: the Next.js app. Better Auth runs under /api/auth there,
//   and the backend verifies tokens against its /api/auth/jwks.
// - WS_URL: the backend's WebSocket server (graph execution events).
// - SIGNUP_ALLOWED: whether the authenticated tests may sign up throwaway
//   accounts when LOAD_TEST_ACCOUNTS is unset (see lib/auth.js).
// - SENTRY: where the target's backend traces land, for perf-compare.
// - PRODUCTION: whether any of the URLs is production's.
//
// K6_BACKEND_URL, K6_FRONTEND_URL and K6_WS_URL override the URLs of any
// target, for a stack on other ports or behind another name. The target's
// name then says which were overridden, and a URL on one of production's
// hosts needs K6_ALLOW_PROD=true whichever target it overrides.
const PRODUCTION_URLS = {
  backend: "https://backend.agpt.co",
  frontend: "https://platform.agpt.co",
  ws: "wss://ws-backend.agpt.co/ws",
};
const PRODUCTION_HOSTS = Object.values(PRODUCTION_URLS).map(hostOf);

const ALLOW_PROD_HINT = "agree it with the team, then set K6_ALLOW_PROD=true";

function hostOf(url) {
  const match = /^[a-z][a-z0-9+.-]*:\/\/([^/?#]*)/i.exec(String(url));
  if (!match) return "";
  return match[1]
    .replace(/^.*@/, "")
    .replace(/:\d*$/, "")
    .replace(/\.$/, "")
    .toLowerCase();
}

function isProductionUrl(url) {
  return PRODUCTION_HOSTS.includes(hostOf(url));
}

function target(name, urls, signupAllowed, sentry) {
  const overridden = [];
  function resolve(variable, fallback) {
    const value = __ENV[variable];
    if (!value) return fallback;
    overridden.push(variable);
    return value;
  }
  const backend = resolve("K6_BACKEND_URL", urls.backend);
  const frontend = resolve("K6_FRONTEND_URL", urls.frontend);
  const ws = resolve("K6_WS_URL", urls.ws);

  const atProduction = [
    ["K6_BACKEND_URL", backend],
    ["K6_FRONTEND_URL", frontend],
    ["K6_WS_URL", ws],
  ].filter(([, url]) => isProductionUrl(url));
  const production = name === "PROD" || atProduction.length > 0;
  if (production && __ENV.K6_ALLOW_PROD !== "true") {
    const what =
      name === "PROD"
        ? "K6_ENVIRONMENT=PROD"
        : `${atProduction.map(([variable]) => variable).join(", ")} on a production host`;
    throw new Error(`${what} sends load at production; ${ALLOW_PROD_HINT}`);
  }

  return {
    NAME:
      overridden.length > 0
        ? `${name} (overridden by ${overridden.join(", ")})`
        : name,
    BACKEND_URL: backend,
    API_BASE_URL: `${backend}/api`,
    FRONTEND_URL: frontend,
    WS_URL: ws,
    SIGNUP_ALLOWED: signupAllowed && !production,
    // A backend somewhere else reports to its own Sentry project, if any.
    SENTRY: overridden.includes("K6_BACKEND_URL") ? null : sentry,
    PRODUCTION: production,
  };
}

export function getEnvironmentConfig() {
  const env = (__ENV.K6_ENVIRONMENT || "PREVIEW").toUpperCase();
  switch (env) {
    case "LOCAL": {
      // k6.sh sets LOCAL_HOST so k6 inside Docker reaches the host's stack.
      const host = __ENV.LOCAL_HOST || "localhost";
      return target(
        "LOCAL",
        {
          backend: `http://${host}:8006`,
          frontend: `http://${host}:3000`,
          ws: `ws://${host}:8001/ws`,
        },
        true,
        null,
      );
    }
    case "DEV":
      return target(
        "DEV",
        {
          backend: "https://dev-server.agpt.co",
          frontend: "https://dev-builder.agpt.co",
          ws: "wss://dev-ws-server.agpt.co/ws",
        },
        false,
        { project: "autogpt-server", environment: "app:dev-behave:cloud" },
      );
    case "PREVIEW":
      return getPreviewConfig(__ENV.PR_NUMBER);
    case "PROD":
      return target("PROD", PRODUCTION_URLS, false, {
        project: "autogpt-server",
        environment: "app:prod-behave:cloud",
      });
    default:
      throw new Error(
        `Unknown K6_ENVIRONMENT "${env}": use PREVIEW, DEV, LOCAL or PROD`,
      );
  }
}

// K6_ENVIRONMENT=PREVIEW PR_NUMBER=<n> targets the `!deploy` preview of PR <n>.
function getPreviewConfig(prNumber) {
  if (!/^[1-9]\d{0,6}$/.test(prNumber || "")) {
    throw new Error(
      "K6_ENVIRONMENT=PREVIEW needs PR_NUMBER=<pull request number>",
    );
  }
  return target(
    `PREVIEW pr-${prNumber}`,
    {
      backend: `https://pr-${prNumber}-server.agpt.co`,
      frontend: `https://autogpt-pr-${prNumber}.vercel.app`,
      ws: `wss://pr-${prNumber}-ws.agpt.co/ws`,
    },
    true,
    { project: "autogpt-server-previews", environment: `pr-${prNumber}` },
  );
}
