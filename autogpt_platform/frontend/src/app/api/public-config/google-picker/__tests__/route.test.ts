import { afterEach, describe, expect, it, vi } from "vitest";

import { GET } from "../route";

const VALID_BROWSER_API_KEY = [
  "AI",
  "za12345678901234567890123456789012345",
].join("");

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("GET /api/public-config/google-picker", () => {
  it("returns validated Picker-scoped runtime values without exposing server keys", async () => {
    vi.stubEnv(
      "GOOGLE_PICKER_CLIENT_ID",
      "123-runtime.apps.googleusercontent.com",
    );
    vi.stubEnv("GOOGLE_PICKER_API_KEY", VALID_BROWSER_API_KEY);
    vi.stubEnv("GOOGLE_PICKER_APP_ID", "1234567890");
    vi.stubEnv("GOOGLE_API_KEY", "must-never-be-public");
    vi.stubEnv("GOOGLE_CLIENT_SECRET", "must-never-be-public");

    const response = await GET();

    expect(response.status).toBe(200);
    expect(response.headers.get("cache-control")).toBe("private, no-store");
    expect(await response.json()).toEqual({
      clientId: "123-runtime.apps.googleusercontent.com",
      developerKey: VALID_BROWSER_API_KEY,
      appId: "1234567890",
    });
  });

  it("falls back to the existing build-time public values", async () => {
    vi.stubEnv("GOOGLE_PICKER_CLIENT_ID", "");
    vi.stubEnv("GOOGLE_PICKER_API_KEY", "");
    vi.stubEnv("GOOGLE_PICKER_APP_ID", "");
    vi.stubEnv(
      "NEXT_PUBLIC_GOOGLE_CLIENT_ID",
      "123-public.apps.googleusercontent.com",
    );
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_API_KEY", VALID_BROWSER_API_KEY);
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_APP_ID", "9876543210");

    const response = await GET();

    expect(response.headers.get("cache-control")).toBe("private, no-store");
    expect(await response.json()).toEqual({
      clientId: "123-public.apps.googleusercontent.com",
      developerKey: VALID_BROWSER_API_KEY,
      appId: "9876543210",
    });
  });

  it("returns null for invalid values instead of reflecting arbitrary environment", async () => {
    vi.stubEnv("GOOGLE_PICKER_CLIENT_ID", "not-a-client-id");
    vi.stubEnv("GOOGLE_PICKER_API_KEY", "server-secret");
    vi.stubEnv("GOOGLE_PICKER_APP_ID", "not-a-project-number");
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_CLIENT_ID", "");
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_API_KEY", "");
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_APP_ID", "");
    vi.stubEnv("DATABASE_URL", "postgresql://secret");

    const response = await GET();

    expect(response.headers.get("cache-control")).toBe("private, no-store");
    expect(await response.json()).toEqual({
      clientId: null,
      developerKey: null,
      appId: null,
    });
  });
});
