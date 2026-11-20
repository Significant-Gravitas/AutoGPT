import Constants from "expo-constants";

function withoutTrailingSlash(url: string) {
  return url.replace(/\/+$/, "");
}

export const FRONTEND_URL = withoutTrailingSlash(
  process.env.EXPO_PUBLIC_FRONTEND_URL || "http://localhost:3000",
);

export const API_URL = withoutTrailingSlash(
  process.env.EXPO_PUBLIC_API_URL || "http://localhost:8006",
);

function ownScheme() {
  const scheme = Constants.expoConfig?.scheme;
  return (Array.isArray(scheme) ? scheme[0] : scheme) || "autogpt-app-dev";
}

/** The link scheme this build registers: one for the store app, one for AutoGPT Dev (app.config.ts). */
export const APP_SCHEME = ownScheme();
