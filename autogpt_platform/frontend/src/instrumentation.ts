import * as Sentry from "@sentry/nextjs";

export async function register() {
  if (process.env.NEXT_RUNTIME === "nodejs") {
    // Next re-raises from here out of server startup, so a TRUSTED_PROXIES entry
    // it cannot parse stops the server instead of the first request that signs in.
    const { isForwardedForTrusted, trustedProxies } = await import(
      "@/lib/auth/client-ip"
    );
    trustedProxies();
    if (
      process.env.TRUST_X_FORWARDED_FOR === "true" &&
      !isForwardedForTrusted()
    ) {
      console.warn(
        "TRUST_X_FORWARDED_FOR is deprecated and ignored; set TRUSTED_PROXIES " +
          "to the proxies in front of the frontend, or the sign-in IP log and " +
          "IP blocks get no client IP",
      );
    }

    await import("../sentry.server.config");
  }

  if (process.env.NEXT_RUNTIME === "edge") {
    await import("../sentry.edge.config");
  }
}

export const onRequestError = Sentry.captureRequestError;
