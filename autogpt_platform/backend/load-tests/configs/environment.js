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
//
// K6_BACKEND_URL, K6_FRONTEND_URL and K6_WS_URL override the URLs of any
// target, for a stack on other ports or behind another name.
function target(name, backendUrl, frontendUrl, wsUrl, signupAllowed, sentry) {
  const backend = __ENV.K6_BACKEND_URL || backendUrl;
  return {
    NAME: name,
    BACKEND_URL: backend,
    API_BASE_URL: `${backend}/api`,
    FRONTEND_URL: __ENV.K6_FRONTEND_URL || frontendUrl,
    WS_URL: __ENV.K6_WS_URL || wsUrl,
    SIGNUP_ALLOWED: signupAllowed,
    SENTRY: sentry,
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
        `http://${host}:8006`,
        `http://${host}:3000`,
        `ws://${host}:8001/ws`,
        true,
        null,
      );
    }
    case "DEV":
      return target(
        "DEV",
        "https://dev-server.agpt.co",
        "https://dev-builder.agpt.co",
        "wss://dev-ws-server.agpt.co/ws",
        false,
        { project: "autogpt-server", environment: "app:dev-behave:cloud" },
      );
    case "PREVIEW":
      return getPreviewConfig(__ENV.PR_NUMBER);
    case "PROD":
      if (__ENV.K6_ALLOW_PROD !== "true") {
        throw new Error(
          "K6_ENVIRONMENT=PROD sends load at production; agree it with the team, then set K6_ALLOW_PROD=true",
        );
      }
      return target(
        "PROD",
        "https://backend.agpt.co",
        "https://platform.agpt.co",
        "wss://ws-backend.agpt.co/ws",
        false,
        { project: "autogpt-server", environment: "app:prod-behave:cloud" },
      );
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
    `https://pr-${prNumber}-server.agpt.co`,
    `https://autogpt-pr-${prNumber}.vercel.app`,
    `wss://pr-${prNumber}-ws.agpt.co/ws`,
    true,
    { project: "autogpt-server-previews", environment: `pr-${prNumber}` },
  );
}
