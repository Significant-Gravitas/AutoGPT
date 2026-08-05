import { afterEach, describe, expect, it, vi } from "vitest";

import { GET } from "../route";

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("GET /api/public-config/google-picker", () => {
  it("returns runtime Google Picker public values without exposing secrets", async () => {
    vi.stubEnv("GOOGLE_CLIENT_ID", "runtime-client-id");
    vi.stubEnv("GOOGLE_API_KEY", "runtime-developer-key");
    vi.stubEnv("GOOGLE_APP_ID", "runtime-app-id");
    vi.stubEnv("GOOGLE_CLIENT_SECRET", "must-never-be-public");

    const response = await GET();

    expect(response.status).toBe(200);
    expect(response.headers.get("cache-control")).toBe("private, no-store");
    expect(await response.json()).toEqual({
      clientId: "runtime-client-id",
      developerKey: "runtime-developer-key",
      appId: "runtime-app-id",
    });
  });

  it("falls back to the existing build-time public values", async () => {
    vi.stubEnv("GOOGLE_CLIENT_ID", "");
    vi.stubEnv("GOOGLE_API_KEY", "");
    vi.stubEnv("GOOGLE_APP_ID", "");
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_CLIENT_ID", "public-client-id");
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_API_KEY", "public-developer-key");
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_APP_ID", "public-app-id");

    const response = await GET();

    expect(await response.json()).toEqual({
      clientId: "public-client-id",
      developerKey: "public-developer-key",
      appId: "public-app-id",
    });
  });

  it("returns null for unconfigured values instead of unrelated environment", async () => {
    vi.stubEnv("GOOGLE_CLIENT_ID", "");
    vi.stubEnv("GOOGLE_API_KEY", "");
    vi.stubEnv("GOOGLE_APP_ID", "");
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_CLIENT_ID", "");
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_API_KEY", "");
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_APP_ID", "");
    vi.stubEnv("DATABASE_URL", "postgresql://secret");

    const response = await GET();

    expect(await response.json()).toEqual({
      clientId: null,
      developerKey: null,
      appId: null,
    });
  });
});
