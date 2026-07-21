import type { User } from "@supabase/supabase-js";
import { describe, expect, it } from "vitest";
import { buildLDContext } from "../helpers";

function userFixture(overrides: Partial<User> = {}): User {
  return {
    id: "00000000-0000-0000-0000-000000000001",
    aud: "authenticated",
    app_metadata: {},
    user_metadata: {},
    created_at: "2026-05-08T12:00:00Z",
    email: "user@example.com",
    role: "authenticated",
    ...overrides,
  } as User;
}

describe("buildLDContext", () => {
  it("returns anonymous context when no user", () => {
    expect(buildLDContext(null)).toEqual({
      kind: "user",
      key: "anonymous",
      anonymous: true,
    });
  });

  it("includes email, email_domain, role, created_at, and custom.role for a full user", () => {
    const ctx = buildLDContext(userFixture());

    expect(ctx).toEqual({
      kind: "user",
      key: "00000000-0000-0000-0000-000000000001",
      anonymous: false,
      email: "user@example.com",
      email_domain: "example.com",
      role: "authenticated",
      created_at: "2026-05-08T12:00:00Z",
      custom: { role: "authenticated" },
    });
  });

  it("preserves the exact created_at string from Supabase (no normalization)", () => {
    const ctx = buildLDContext(
      userFixture({ created_at: "2025-01-15T08:30:45.123456Z" }),
    );

    expect("created_at" in ctx && ctx.created_at).toBe(
      "2025-01-15T08:30:45.123456Z",
    );
  });

  it("omits created_at when missing — falsy spread skips the key", () => {
    const ctx = buildLDContext(
      userFixture({ created_at: undefined as unknown as string }),
    );

    expect(ctx).not.toHaveProperty("created_at");
    expect("email" in ctx && ctx.email).toBe("user@example.com");
  });

  it("omits role and leaves custom empty when role missing", () => {
    const ctx = buildLDContext(userFixture({ role: undefined }));

    expect(ctx).not.toHaveProperty("role");
    expect("custom" in ctx && ctx.custom).toEqual({});
  });

  it("omits email and email_domain when email missing", () => {
    const ctx = buildLDContext(userFixture({ email: undefined }));

    expect(ctx).not.toHaveProperty("email");
    expect(ctx).not.toHaveProperty("email_domain");
  });

  it("derives email_domain from the last @ segment of email", () => {
    const ctx = buildLDContext(userFixture({ email: "a.b@sub.example.com" }));

    expect("email_domain" in ctx && ctx.email_domain).toBe("sub.example.com");
  });
});
