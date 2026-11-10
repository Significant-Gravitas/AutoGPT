function withoutTrailingSlash(url: string) {
  return url.replace(/\/+$/, "");
}

export const FRONTEND_URL = withoutTrailingSlash(
  process.env.EXPO_PUBLIC_FRONTEND_URL || "http://localhost:3000",
);

export const API_URL = withoutTrailingSlash(
  process.env.EXPO_PUBLIC_API_URL || "http://localhost:8006",
);

export const APP_SCHEME = "autogpt-app";
