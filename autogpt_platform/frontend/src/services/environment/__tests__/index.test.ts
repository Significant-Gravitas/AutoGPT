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

describe("Sentry environment tagging", () => {
  function stubDeployment(appEnv: string, vercelEnv: string) {
    vi.stubEnv("NEXT_PUBLIC_APP_ENV", appEnv);
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "CLOUD");
    vi.stubEnv("NEXT_PUBLIC_VERCEL_ENV", vercelEnv);
  }

  it("tags a Vercel preview as preview, not as the app env it inherits", () => {
    stubDeployment("prod", "preview");

    expect(environment.getEnvironmentStr()).toBe("app:preview-behave:cloud");
  });

  it("tags the production deployment as prod", () => {
    stubDeployment("prod", "production");

    expect(environment.getEnvironmentStr()).toBe("app:prod-behave:cloud");
  });

  it("keeps the dev deployment distinct from production", () => {
    stubDeployment("dev", "production");

    expect(environment.getEnvironmentStr()).toBe("app:dev-behave:cloud");
  });

  it("tags a self-hosted instance by its own env", () => {
    vi.stubEnv("NEXT_PUBLIC_APP_ENV", "local");
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "LOCAL");
    vi.stubEnv("NEXT_PUBLIC_VERCEL_ENV", "");

    expect(environment.getEnvironmentStr()).toBe("app:local-behave:local");
  });

  it("detects a preview from the server-only VERCEL_ENV as well", () => {
    vi.stubEnv("NEXT_PUBLIC_VERCEL_ENV", "");
    vi.stubEnv("VERCEL_ENV", "preview");

    expect(environment.isVercelPreview()).toBe(true);
  });
});

describe("Sentry enablement", () => {
  it("stays off under the test runner, where no cloud deployment env is set", () => {
    expect(environment.isSentryEnabled()).toBe(false);
  });

  it("is on for a cloud deployment", () => {
    vi.stubEnv("NEXT_PUBLIC_APP_ENV", "prod");
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "CLOUD");

    expect(environment.isSentryEnabled()).toBe(true);
  });

  it("is off for a self-hosted instance", () => {
    vi.stubEnv("NEXT_PUBLIC_APP_ENV", "prod");
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "LOCAL");

    expect(environment.isSentryEnabled()).toBe(false);
  });

  it("honours the DISABLE_SENTRY kill switch on a cloud deployment", () => {
    vi.stubEnv("NEXT_PUBLIC_APP_ENV", "prod");
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "CLOUD");
    vi.stubEnv("DISABLE_SENTRY", "true");

    expect(environment.isSentryEnabled()).toBe(false);
  });
});
