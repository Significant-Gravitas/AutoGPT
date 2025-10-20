import { headers } from "next/headers";

export enum BehaveAs {
  CLOUD = "CLOUD",
  LOCAL = "LOCAL",
}

function getBehaveAs(): BehaveAs {
  return process.env.NEXT_PUBLIC_BEHAVE_AS === "CLOUD"
    ? BehaveAs.CLOUD
    : BehaveAs.LOCAL;
}

export enum AppEnv {
  LOCAL = "local",
  DEV = "dev",
  PROD = "prod",
}

function getAppEnv(): AppEnv {
  const env = process.env.NEXT_PUBLIC_APP_ENV;
  if (env === "dev") return AppEnv.DEV;
  if (env === "prod") return AppEnv.PROD;
  // Some places use prod and others production
  if (env === "production") return AppEnv.PROD;
  return AppEnv.LOCAL;
}

function getAnalyticsWebsiteId() {
  return process.env.NEXT_PUBLIC_ANALYTICS_WEBSITE_ID;
}

function getAGPTServerApiUrl() {
  if (environment.isServerSide() && process.env.AGPT_SERVER_URL) {
    return process.env.AGPT_SERVER_URL;
  }

  return process.env.NEXT_PUBLIC_AGPT_SERVER_URL || "http://localhost:8006/api";
}

function getAGPTServerBaseUrl() {
  return getAGPTServerApiUrl().replace("/api", "");
}

function getAGPTWsServerUrl() {
  if (environment.isServerSide() && process.env.AGPT_WS_SERVER_URL) {
    return process.env.AGPT_WS_SERVER_URL;
  }

  return process.env.NEXT_PUBLIC_AGPT_WS_SERVER_URL || "ws://localhost:8001/ws";
}

function getSupabaseUrl() {
  if (environment.isServerSide() && process.env.SUPABASE_URL) {
    return process.env.SUPABASE_URL;
  }

  return process.env.NEXT_PUBLIC_SUPABASE_URL || "http://localhost:8000";
}

function getSupabaseAnonKey() {
  return process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || "";
}

function getEnvironmentStr() {
  return `app:${getAppEnv().toLowerCase()}-behave:${getBehaveAs().toLowerCase()}`;
}

function isProd() {
  return process.env.NODE_ENV === "production";
}

function isDev() {
  return process.env.NODE_ENV === "development";
}

function isCloud() {
  return getBehaveAs() === BehaveAs.CLOUD;
}

function isLocal() {
  return getBehaveAs() === BehaveAs.LOCAL;
}

function isServerSide() {
  return typeof window === "undefined";
}

function isClientSide() {
  return typeof window !== "undefined";
}

function isCAPTCHAEnabled() {
  return process.env.NEXT_PUBLIC_TURNSTILE === "enabled";
}

function areFeatureFlagsEnabled() {
  return process.env.NEXT_PUBLIC_LAUNCHDARKLY_ENABLED === "enabled";
}

async function areAnalyticsEnabled() {
  const withWebsiteId = getAnalyticsWebsiteId();
  if (!withWebsiteId) {
    return false;
  }

  if (isClientSide()) {
    return window.location.hostname.includes(withWebsiteId);
  }

  const headersList = await headers();
  const host = headersList.get("host") || "";
  return host.includes("platform.agpt.co");
}

export const environment = {
  // Generic
  getEnvironmentStr,
  // Get environment variables config
  getBehaveAs,
  getAppEnv,
  getAGPTServerApiUrl,
  getAGPTServerBaseUrl,
  getAGPTWsServerUrl,
  getSupabaseUrl,
  getSupabaseAnonKey,
  getAnalyticsWebsiteId,
  // Assertions
  isServerSide,
  isClientSide,
  isProd,
  isDev,
  isCloud,
  isLocal,
  isCAPTCHAEnabled,
  areAnalyticsEnabled,
  areFeatureFlagsEnabled,
};
