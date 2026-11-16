import * as Sentry from "@sentry/nextjs";

export async function register() {
  if (process.env.NEXT_RUNTIME === "nodejs") {
    // Next re-raises from here out of server startup, so a TRUSTED_PROXIES entry
    // it cannot parse stops the server instead of the first request that signs in.
    const { trustedProxies } = await import("@/lib/auth/client-ip");
    trustedProxies();

    await import("../sentry.server.config");
  }

  if (process.env.NEXT_RUNTIME === "edge") {
    await import("../sentry.edge.config");
  }
}

export const onRequestError = Sentry.captureRequestError;
