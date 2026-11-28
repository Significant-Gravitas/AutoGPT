import * as Sentry from "@sentry/react-native";

const DSN = process.env.EXPO_PUBLIC_SENTRY_DSN;

/**
 * Crash and error reporting, in store and dev builds only: never while
 * developing, and never without a DSN. The SDK names each event's release
 * after the app's version and build number, and tags any over-the-air update.
 */
export function startMonitoring() {
  if (!DSN || __DEV__) return;
  Sentry.init({
    dsn: DSN,
    environment: process.env.EXPO_PUBLIC_SENTRY_ENVIRONMENT || "production",
    sendDefaultPii: false,
    // Console lines and tap labels can hold what people wrote or who they
    // talk to; the stack trace and requests are enough to find a crash.
    enableUserInteractionTracing: false,
    beforeBreadcrumb: (breadcrumb) =>
      breadcrumb.category === "console" || breadcrumb.category === "touch" ? null : breadcrumb,
  });
}
