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

  it.each([
    ["http://localhost:8006/api", "http://localhost:8006"],
    ["https://api.example.com/api", "https://api.example.com"],
    ["https://example.com/apis/api", "https://example.com/apis"],
    ["https://api.example.com/api/", "https://api.example.com"],
    ["https://backend.example.com", "https://backend.example.com"],
  ])("strips only a trailing /api from %s", (apiUrl, baseUrl) => {
    vi.stubGlobal("window", undefined);
    vi.stubEnv("AGPT_SERVER_URL", apiUrl);

    expect(environment.getAGPTServerBaseUrl()).toBe(baseUrl);
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
  function stubDeployment(
    appEnv: string,
    vercelEnv: string,
    targetEnv?: string,
  ) {
    vi.stubEnv("NEXT_PUBLIC_APP_ENV", appEnv);
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "CLOUD");
    vi.stubEnv("NEXT_PUBLIC_VERCEL_ENV", vercelEnv);
    vi.stubEnv("NEXT_PUBLIC_VERCEL_TARGET_ENV", targetEnv);
    vi.stubEnv("VERCEL_TARGET_ENV", undefined);
  }

  it("tags a Vercel preview as preview, not as the app env it inherits", () => {
    stubDeployment("prod", "preview", "preview");

    expect(environment.getEnvironmentStr()).toBe("app:preview-behave:cloud");
  });

  it.each([
    ["unset", undefined],
    ["empty", ""],
  ])(
    "tags a preview by VERCEL_ENV alone when the target environment is %s",
    (_, targetEnv) => {
      stubDeployment("prod", "preview", targetEnv);

      expect(environment.getEnvironmentStr()).toBe("app:preview-behave:cloud");
    },
  );

  it("tags the production deployment as prod", () => {
    stubDeployment("prod", "production", "production");

    expect(environment.getEnvironmentStr()).toBe("app:prod-behave:cloud");
  });

  it("keeps the dev deployment distinct from production", () => {
    stubDeployment("dev", "production");

    expect(environment.getEnvironmentStr()).toBe("app:dev-behave:cloud");
  });

  it("tags the dev deployment as dev although Vercel calls its staging environment a preview", () => {
    stubDeployment("dev", "preview", "staging");

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

  it("tags events with the environment a deployment names for itself", () => {
    stubDeployment("prod", "preview", "preview");
    vi.stubEnv("NEXT_PUBLIC_SENTRY_ENVIRONMENT", "pr-123");

    expect(environment.getSentryEnvironment()).toBe("pr-123");
  });

  it("ignores whitespace around the environment a deployment names", () => {
    stubDeployment("prod", "preview");
    vi.stubEnv("NEXT_PUBLIC_SENTRY_ENVIRONMENT", "  pr-123 ");

    expect(environment.getSentryEnvironment()).toBe("pr-123");
  });

  it.each([
    ["unset", undefined],
    ["empty", ""],
    ["blank", "   "],
  ])(
    "keeps the computed environment when the override is %s",
    (_, override) => {
      stubDeployment("prod", "preview");
      vi.stubEnv("NEXT_PUBLIC_SENTRY_ENVIRONMENT", override);

      expect(environment.getSentryEnvironment()).toBe(
        "app:preview-behave:cloud",
      );
      expect(environment.getSentryEnvironment()).toBe(
        environment.getEnvironmentStr(),
      );
    },
  );
});

describe("Vercel preview detection", () => {
  const deployments = [
    {
      scenario: "a branch preview",
      vercelEnv: "preview",
      targetEnv: "preview",
      isPreview: true,
    },
    {
      scenario: "a custom environment such as staging",
      vercelEnv: "preview",
      targetEnv: "staging",
      isPreview: false,
    },
    {
      scenario: "a build with no target environment",
      vercelEnv: "preview",
      targetEnv: undefined,
      isPreview: true,
    },
    {
      scenario: "a build with an empty target environment",
      vercelEnv: "preview",
      targetEnv: "",
      isPreview: true,
    },
    {
      scenario: "production",
      vercelEnv: "production",
      targetEnv: "production",
      isPreview: false,
    },
    {
      scenario: "local development on Vercel",
      vercelEnv: "development",
      targetEnv: "development",
      isPreview: false,
    },
    {
      scenario: "a build outside Vercel",
      vercelEnv: undefined,
      targetEnv: undefined,
      isPreview: false,
    },
  ];

  describe.each([
    ["the browser bundle", "NEXT_PUBLIC_"],
    ["the server", ""],
  ])("as read by %s", (_, prefix) => {
    it.each(deployments)(
      "reports $scenario as preview: $isPreview",
      ({ vercelEnv, targetEnv, isPreview }) => {
        vi.stubEnv("NEXT_PUBLIC_VERCEL_ENV", undefined);
        vi.stubEnv("VERCEL_ENV", undefined);
        vi.stubEnv("NEXT_PUBLIC_VERCEL_TARGET_ENV", undefined);
        vi.stubEnv("VERCEL_TARGET_ENV", undefined);
        vi.stubEnv(`${prefix}VERCEL_ENV`, vercelEnv);
        vi.stubEnv(`${prefix}VERCEL_TARGET_ENV`, targetEnv);

        expect(environment.isVercelPreview()).toBe(isPreview);
      },
    );
  });
});

describe("Sentry enablement", () => {
  const HOSTED_DSN_PROJECT = "/4507946746380288";

  it("stays off under the test runner, where no cloud deployment env is set", () => {
    expect(environment.isSentryEnabled()).toBe(false);
  });

  it("reports to AutoGPT's Sentry from a hosted deployment on Vercel", () => {
    vi.stubEnv("NEXT_PUBLIC_APP_ENV", "prod");
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "CLOUD");
    vi.stubEnv("NEXT_PUBLIC_VERCEL_ENV", "production");

    expect(environment.isSentryEnabled()).toBe(true);
    expect(environment.getSentryDSN()).toContain(HOSTED_DSN_PROJECT);
  });

  it("reports from a hosted preview deployment", () => {
    vi.stubEnv("NEXT_PUBLIC_APP_ENV", "dev");
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "CLOUD");
    vi.stubEnv("NEXT_PUBLIC_VERCEL_ENV", "preview");

    expect(environment.isSentryEnabled()).toBe(true);
  });

  it("is off for a self-hosted instance", () => {
    vi.stubEnv("NEXT_PUBLIC_APP_ENV", "prod");
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "LOCAL");

    expect(environment.isSentryEnabled()).toBe(false);
  });

  it("does not report to AutoGPT's Sentry from a self-hosted build configured like the cloud", () => {
    vi.stubEnv("NEXT_PUBLIC_APP_ENV", "prod");
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "CLOUD");
    vi.stubEnv("NEXT_PUBLIC_VERCEL_ENV", "");
    vi.stubEnv("VERCEL", "");

    expect(environment.isSentryEnabled()).toBe(false);
  });

  it("reports to a self-hoster's own Sentry once they opt in with a DSN", () => {
    vi.stubEnv("NEXT_PUBLIC_SENTRY_DSN", "https://key@sentry.example.com/7");

    expect(environment.getSentryDSN()).toBe("https://key@sentry.example.com/7");
  });

  it("honours the DISABLE_SENTRY kill switch on a cloud deployment", () => {
    vi.stubEnv("NEXT_PUBLIC_APP_ENV", "prod");
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "CLOUD");
    vi.stubEnv("NEXT_PUBLIC_VERCEL_ENV", "production");
    vi.stubEnv("DISABLE_SENTRY", "true");

    expect(environment.isSentryEnabled()).toBe(false);
  });
});
