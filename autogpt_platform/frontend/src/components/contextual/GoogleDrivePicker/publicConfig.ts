import type { EnvironmentDrivenGoogleConfig } from "./types";

export const GOOGLE_PICKER_PUBLIC_CONFIG_QUERY_KEY = [
  "public-config",
  "google-picker",
] as const;

export async function fetchGooglePickerPublicConfig(): Promise<EnvironmentDrivenGoogleConfig> {
  try {
    const response = await fetch("/api/public-config/google-picker", {
      cache: "no-store",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return parseGooglePickerPublicConfig(await response.json());
  } catch (error) {
    throw new Error("Failed to load Google Picker runtime configuration.", {
      cause: error,
    });
  }
}

export function resolveGooglePickerConfig(
  preferred: EnvironmentDrivenGoogleConfig,
  fallback: EnvironmentDrivenGoogleConfig,
): EnvironmentDrivenGoogleConfig {
  return {
    clientId: preferred.clientId || fallback.clientId,
    developerKey: preferred.developerKey || fallback.developerKey,
    appId: preferred.appId || fallback.appId,
  };
}

export function hasCompleteGooglePickerConfig(
  config: EnvironmentDrivenGoogleConfig,
): config is Required<EnvironmentDrivenGoogleConfig> {
  return Boolean(config.clientId && config.developerKey && config.appId);
}

function parseGooglePickerPublicConfig(
  value: unknown,
): EnvironmentDrivenGoogleConfig {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Google Picker public config response is invalid");
  }
  const config = value as Record<string, unknown>;
  return {
    clientId: readOptionalString(config.clientId, "clientId"),
    developerKey: readOptionalString(config.developerKey, "developerKey"),
    appId: readOptionalString(config.appId, "appId"),
  };
}

function readOptionalString(value: unknown, field: string) {
  if (value === null || value === undefined || value === "") return undefined;
  if (typeof value !== "string") {
    throw new Error(`Google Picker public config ${field} is invalid`);
  }
  return value;
}
