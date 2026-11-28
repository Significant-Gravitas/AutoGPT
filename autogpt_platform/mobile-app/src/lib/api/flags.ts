import { apiRequest } from "./client";

/** `FrontendFlagValues`: a flag left out, or null, has no answer and stays off. */
export interface FlagValues {
  flags: Record<string, unknown>;
  configured: boolean;
}

export const flagKeys = {
  all: ["feature-flags"] as const,
};

export function getFlags() {
  return apiRequest<FlagValues>("/api/feature-flags");
}

export function isFlagOn(data: FlagValues | undefined, key: string) {
  return data?.flags[key] === true;
}
