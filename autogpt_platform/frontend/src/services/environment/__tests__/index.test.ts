import { afterEach, describe, expect, it, vi } from "vitest";

import { environment } from "../index";

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("AutoGPT server URL resolution", () => {
  it("resolves a relative browser API URL against an HTTP origin", () => {
    vi.stubGlobal("window", {
      location: { origin: "http://autogpt.local:3000" },
    });
    vi.stubEnv("NEXT_PUBLIC_AGPT_SERVER_URL", "/backend/api");

    expect(environment.getAGPTServerApiUrl()).toBe(
      "http://autogpt.local:3000/backend/api",
    );
    expect(environment.getAGPTServerBaseUrl()).toBe(
      "http://autogpt.local:3000/backend",
    );
  });

  it("resolves a relative browser API URL against an HTTPS origin", () => {
    vi.stubGlobal("window", {
      location: { origin: "https://autogpt.example.com" },
    });
    vi.stubEnv("NEXT_PUBLIC_AGPT_SERVER_URL", "/backend/api");

    expect(environment.getAGPTServerApiUrl()).toBe(
      "https://autogpt.example.com/backend/api",
    );
  });

  it("resolves a relative browser WebSocket URL using ws over HTTP", () => {
    vi.stubGlobal("window", {
      location: { origin: "http://autogpt.local:3000" },
    });
    vi.stubEnv("NEXT_PUBLIC_AGPT_WS_SERVER_URL", "/backend/ws");

    expect(environment.getAGPTWsServerUrl()).toBe(
      "ws://autogpt.local:3000/backend/ws",
    );
  });

  it("resolves a relative browser WebSocket URL using wss over HTTPS", () => {
    vi.stubGlobal("window", {
      location: { origin: "https://autogpt.example.com" },
    });
    vi.stubEnv("NEXT_PUBLIC_AGPT_WS_SERVER_URL", "/backend/ws");

    expect(environment.getAGPTWsServerUrl()).toBe(
      "wss://autogpt.example.com/backend/ws",
    );
  });

  it("leaves explicit server-side backend URLs untouched", () => {
    vi.stubGlobal("window", undefined);
    vi.stubEnv("AGPT_SERVER_URL", "http://backend.internal:8006/api");
    vi.stubEnv("AGPT_WS_SERVER_URL", "ws://backend.internal:8001/ws");
    vi.stubEnv("NEXT_PUBLIC_AGPT_SERVER_URL", "/backend/api");
    vi.stubEnv("NEXT_PUBLIC_AGPT_WS_SERVER_URL", "/backend/ws");

    expect(environment.getAGPTServerApiUrl()).toBe(
      "http://backend.internal:8006/api",
    );
    expect(environment.getAGPTWsServerUrl()).toBe(
      "ws://backend.internal:8001/ws",
    );
  });
});
