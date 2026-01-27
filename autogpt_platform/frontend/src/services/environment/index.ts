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

function getPreviewStealingDev() {
  const branch = process.env.NEXT_PUBLIC_PREVIEW_STEALING_DEV || "";
  const appEnv = getAppEnv();

  if (
    !branch ||
    branch === "dev" ||
    branch === "refs/heads/dev" ||
    appEnv !== AppEnv.DEV
  ) {
    return null;
  }

  return branch;
}

function getPostHogCredentials() {
  return {
    key: process.env.NEXT_PUBLIC_POSTHOG_KEY,
    host: process.env.NEXT_PUBLIC_POSTHOG_HOST,
  };
}

function isProductionBuild() {
  return process.env.NODE_ENV === "production";
}

function isDevelopmentBuild() {
  return process.env.NODE_ENV === "development";
}

function isDev() {
  return isCloud() && getAppEnv() === AppEnv.DEV;
}

function isProd() {
  return isCloud() && getAppEnv() === AppEnv.PROD;
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

function isVercelPreview() {
  return process.env.VERCEL_ENV === "preview";
}

function areFeatureFlagsEnabled() {
  return process.env.NEXT_PUBLIC_LAUNCHDARKLY_ENABLED === "enabled";
}

function isPostHogEnabled() {
  const inCloud = isCloud();
  const key = process.env.NEXT_PUBLIC_POSTHOG_KEY;
  const host = process.env.NEXT_PUBLIC_POSTHOG_HOST;
  return inCloud && key && host;
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
  getPreviewStealingDev,
  getPostHogCredentials,
  // Assertions
  isServerSide,
  isClientSide,
  isProductionBuild,
  isDevelopmentBuild,
  isDev,
  isProd,
  isCloud,
  isLocal,
  isVercelPreview,
  isPostHogEnabled,
  areFeatureFlagsEnabled,
};
