import { describe, expect, it } from "vitest";
import { localExecutorDeployment } from "../helpers";

describe("localExecutorDeployment", () => {
  it("uses the configured backend and this frontend for authorization", () => {
    const deployment = localExecutorDeployment(
      "https://api.example.com/api",
      "https://app.example.com",
    );
    expect(deployment.authCommand).toBe(
      "autogpt-shim --platform-url 'https://api.example.com' --platform-oauth-url 'https://app.example.com/auth' auth",
    );
    expect(deployment.startCommand).toBe(
      "autogpt-shim --platform-url 'https://api.example.com' start",
    );
    expect(deployment.config).toBe(
      'platform_url = "https://api.example.com"\nplatform_oauth_url = "https://app.example.com/auth"',
    );
  });

  it("preserves a self-hosted backend path prefix", () => {
    expect(
      localExecutorDeployment("/backend/api/", "https://my-autogpt.example")
        .platformURL,
    ).toBe("https://my-autogpt.example/backend");
  });

  it("quotes command arguments without exposing URL shell metacharacters", () => {
    expect(
      localExecutorDeployment(
        "https://api.example.com/it's/api",
        "https://app.example.com",
      ).startCommand,
    ).toBe(
      "autogpt-shim --platform-url 'https://api.example.com/it%27s' start",
    );
  });
});
