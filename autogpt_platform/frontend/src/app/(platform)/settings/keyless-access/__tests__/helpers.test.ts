import { describe, expect, test } from "vitest";

import { APIKeyPermission } from "@/app/api/__generated__/models/aPIKeyPermission";
import type { WorkloadIdentityInfo } from "@/app/api/__generated__/models/workloadIdentityInfo";
import { WorkloadIdentityStatus } from "@/app/api/__generated__/models/workloadIdentityStatus";

import { problemsWith } from "../components/CreateWorkloadDialog/schema";
import {
  callSnippet,
  exchangeSnippet,
  siteSetupSnippets,
  tokenStep,
} from "../components/UsageGuide/helpers";
import {
  buildWorkloadSpec,
  EMPTY_FORM_VALUES,
  exampleEndpoint,
  platformOf,
  type WorkloadFormValues,
} from "../helpers";

const AUDIENCE = "https://backend.agpt.co";

function values(overrides: Partial<WorkloadFormValues>): WorkloadFormValues {
  return {
    ...EMPTY_FORM_VALUES,
    permissions: [APIKeyPermission.IDENTITY],
    ...overrides,
  };
}

function identity(
  overrides: Partial<WorkloadIdentityInfo> = {},
): WorkloadIdentityInfo {
  return {
    id: "wi-123",
    name: "deploy",
    issuer: "https://token.actions.githubusercontent.com",
    audience: AUDIENCE,
    claim_conditions: [{ claim: "repository", pattern: "acme/api" }],
    permissions: [APIKeyPermission.IDENTITY],
    token_ttl_seconds: 900,
    status: WorkloadIdentityStatus.ACTIVE,
    created_at: new Date("2026-10-22T00:00:00Z"),
    user_id: "user-1",
    ...overrides,
  };
}

describe("buildWorkloadSpec", () => {
  test("pins a GitHub workload to its repository, branch and environment", () => {
    const spec = buildWorkloadSpec(
      values({
        platform: "github",
        repository: "acme/api",
        branch: "main",
        environment: "production",
      }),
      AUDIENCE,
    );

    expect(spec.issuer).toBe("https://token.actions.githubusercontent.com");
    expect(spec.audience).toBe(AUDIENCE);
    expect(spec.claim_conditions).toEqual([
      { claim: "repository", pattern: "acme/api" },
      { claim: "ref", pattern: "refs/heads/main" },
      { claim: "environment", pattern: "production" },
    ]);
    expect(spec.name).toBe("GitHub Actions · acme/api");
  });

  test("leaves out optional GitHub conditions that were left empty", () => {
    const spec = buildWorkloadSpec(
      values({ platform: "github", repository: "acme/api" }),
      AUDIENCE,
    );

    expect(spec.claim_conditions).toEqual([
      { claim: "repository", pattern: "acme/api" },
    ]);
  });

  test("uses the team issuer for Vercel and pins team, project and environment", () => {
    const spec = buildWorkloadSpec(
      values({ platform: "vercel", team: "acme", project: "acme-website" }),
      AUDIENCE,
    );

    expect(spec.issuer).toBe("https://oidc.vercel.com/acme");
    expect(spec.claim_conditions).toEqual([
      { claim: "owner", pattern: "acme" },
      { claim: "project", pattern: "acme-website" },
      { claim: "environment", pattern: "production" },
    ]);
  });

  test("uses the global Vercel issuer when the project does", () => {
    const spec = buildWorkloadSpec(
      values({
        platform: "vercel",
        team: "acme",
        project: "web",
        vercelIssuerMode: "global",
        vercelEnvironment: "any",
      }),
      AUDIENCE,
    );

    expect(spec.issuer).toBe("https://oidc.vercel.com");
    expect(spec.claim_conditions).toEqual([
      { claim: "owner", pattern: "acme" },
      { claim: "project", pattern: "web" },
    ]);
  });

  test("makes your own site the issuer, without a trailing slash", () => {
    const spec = buildWorkloadSpec(
      values({
        platform: "website",
        siteUrl: "https://acme.com/",
        subject: "backend",
      }),
      AUDIENCE,
    );

    expect(spec.issuer).toBe("https://acme.com");
    expect(spec.audience).toBe(AUDIENCE);
    expect(spec.claim_conditions).toEqual([
      { claim: "sub", pattern: "backend" },
    ]);
    expect(spec.name).toBe("acme.com · backend");
  });

  test("lets a custom provider keep AutoGPT's audience when none is given", () => {
    const spec = buildWorkloadSpec(
      values({
        platform: "custom",
        issuer: "https://k8s.example.com",
        claim: "sub",
        pattern: "system:serviceaccount:prod:agent",
      }),
      AUDIENCE,
    );

    expect(spec.audience).toBe(AUDIENCE);
    expect(spec.claim_conditions).toEqual([
      { claim: "sub", pattern: "system:serviceaccount:prod:agent" },
    ]);
  });
});

describe("problemsWith", () => {
  test.each([
    [{ platform: "github", repository: "" }, "repository"],
    [{ platform: "github", repository: "acme" }, "repository"],
    [{ platform: "github", repository: "acme/api", branch: "*" }, "branch"],
    [{ platform: "vercel", team: "Acme", project: "web" }, "team"],
    [{ platform: "google", serviceAccount: "not-an-email" }, "serviceAccount"],
    [
      { platform: "website", siteUrl: "http://acme.com", subject: "x" },
      "siteUrl",
    ],
    [
      { platform: "website", siteUrl: "https://acme.com", subject: "" },
      "subject",
    ],
    [
      {
        platform: "custom",
        issuer: "https://k.example",
        claim: "aud",
        pattern: "x",
      },
      "claim",
    ],
    [
      {
        platform: "custom",
        issuer: "https://k.example",
        claim: "sub",
        pattern: "*",
      },
      "pattern",
    ],
  ] as [Partial<WorkloadFormValues>, keyof WorkloadFormValues][])(
    "flags %o on %s",
    (overrides, field) => {
      const fields = problemsWith(values(overrides)).map(([name]) => name);
      expect(fields).toContain(field);
    },
  );

  test("accepts a complete GitHub workload", () => {
    expect(
      problemsWith(values({ platform: "github", repository: "acme/api" })),
    ).toEqual([]);
  });
});

describe("usage snippets", () => {
  test("detects the platform from the issuer", () => {
    expect(platformOf(identity())).toBe("github");
    expect(
      platformOf(identity({ issuer: "https://oidc.vercel.com/acme" })),
    ).toBe("vercel");
    expect(platformOf(identity({ issuer: "https://acme.com" }))).toBe("custom");
  });

  test("asks GitHub for a token meant for the identity's audience", () => {
    const step = tokenStep({ identity: identity(), platform: "github" });
    const code = step.snippets.map((s) => s.code).join("\n");

    expect(code).toContain("id-token: write");
    expect(code).toContain(`&audience=${AUDIENCE}`);
  });

  test("exchanges at the token endpoint naming the workload", () => {
    const code = exchangeSnippet("wi-123", AUDIENCE);

    expect(code).toContain(`curl -s ${AUDIENCE}/api/oauth/token`);
    expect(code).toContain('WORKLOAD_ID="wi-123"');
    expect(code).toContain("urn:ietf:params:oauth:grant-type:token-exchange");
    expect(code).toContain('\\"subject_token\\": \\"$OIDC_TOKEN\\"');
  });

  test("calls the API with the token as a bearer, on an endpoint it may use", () => {
    expect(callSnippet(identity(), AUDIENCE)).toBe(
      `curl ${AUDIENCE}/external-api/v1/me \\\n  -H "Authorization: Bearer $AUTOGPT_TOKEN"`,
    );
    expect(exampleEndpoint([APIKeyPermission.READ_BLOCK])).toBe(
      "/external-api/v1/blocks",
    );
  });

  test("tells your own site exactly what to publish", () => {
    const site = identity({
      issuer: "https://acme.com",
      claim_conditions: [{ claim: "sub", pattern: "backend" }],
    });
    const setup = siteSetupSnippets(site);
    const step = tokenStep({ identity: site, platform: "website" });

    expect(setup.discoveryUrl).toBe(
      "https://acme.com/.well-known/openid-configuration",
    );
    expect(JSON.parse(setup.discovery)).toEqual({
      issuer: "https://acme.com",
      jwks_uri: "https://acme.com/.well-known/jwks.json",
    });
    expect(step.snippets[0].code).toContain('.setIssuer("https://acme.com")');
    expect(step.snippets[0].code).toContain('.setSubject("backend")');
    expect(step.snippets[0].code).toContain(`.setAudience("${AUDIENCE}")`);
  });
});
