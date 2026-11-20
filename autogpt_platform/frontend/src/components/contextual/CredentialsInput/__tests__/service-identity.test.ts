import { describe, expect, it } from "vitest";
import type { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api/types";
import { credentialService } from "../service-identity";

function schema(url: string): BlockIOCredentialsSubSchema {
  return {
    type: "object",
    properties: {},
    credentials_provider: ["http"],
    credentials_types: ["host_scoped"],
    discriminator_values: [url],
  };
}

describe("credentialService", () => {
  it("names known API hosts", () => {
    expect(
      credentialService(schema("https://api.linear.app/graphql")),
    ).toMatchObject({ name: "Linear", icon: "linear", bearer: false });
    expect(
      credentialService(schema("https://api.github.com/repos")),
    ).toMatchObject({ name: "GitHub", icon: "github", bearer: true });
  });

  it("shows the exact host for unknown services without using a trusted app's name", () => {
    for (const host of [
      "api.linear.app.evil.example",
      "api.linear.app:8443",
      "api.other.example",
    ]) {
      expect(credentialService(schema(`https://${host}/path`))).toEqual({
        host,
      });
    }
  });

  it("does not change native providers or infer an app without one host", () => {
    expect(
      credentialService({
        ...schema("https://api.github.com"),
        credentials_provider: ["github"],
      }),
    ).toBeUndefined();
    expect(
      credentialService({
        ...schema("https://api.github.com"),
        discriminator_values: [],
      }),
    ).toBeUndefined();
    expect(
      credentialService({
        ...schema("https://api.github.com"),
        discriminator_values: [
          "https://api.github.com",
          "https://api.linear.app",
        ],
      }),
    ).toBeUndefined();
  });
});
