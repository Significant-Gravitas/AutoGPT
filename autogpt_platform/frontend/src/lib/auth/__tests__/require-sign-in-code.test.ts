// @vitest-environment node
import type { GenericEndpointContext } from "@better-auth/core";
import type { ValidateUserInfoSource } from "better-auth";
import { describe, expect, it } from "vitest";
import { requireSignInCode } from "../require-sign-in-code";

// What the hook decides before it reaches the database: a context it would
// have to use throws.
const untouchable = new Proxy({} as GenericEndpointContext, {
  get() {
    throw new Error("the hook should not need the request for this");
  },
});

function linking(
  providerId: string,
  user: { email: string; emailVerified: boolean },
) {
  const source: ValidateUserInfoSource = {
    method: "oauth",
    action: "link-account",
    oauth: { providerId, profile: { email: user.email } },
  };
  return requireSignInCode({ user, source }, untouchable);
}

describe("requireSignInCode", () => {
  it("leaves Apple to Better Auth's rule", async () => {
    expect(
      await linking("apple", {
        email: "pat@company.example",
        emailVerified: true,
      }),
    ).toBeUndefined();
  });

  it.each(["google", "microsoft", "github", "discord"])(
    "refuses %s's unverified address without emailing a code",
    async (provider) => {
      expect(
        await linking(provider, {
          email: "pat@company.example",
          emailVerified: false,
        }),
      ).toEqual({ error: "account_not_linked" });
    },
  );

  it("lets a Gmail address through Google join at once", async () => {
    expect(
      await linking("google", { email: "pat@gmail.com", emailVerified: true }),
    ).toBeUndefined();
  });

  it("asks nothing of a sign-up or a returning sign-in", async () => {
    for (const action of ["create-user", "sign-in"] as const) {
      const source: ValidateUserInfoSource = {
        method: "oauth",
        action,
        oauth: { providerId: "github", profile: {} },
      };
      expect(
        await requireSignInCode(
          {
            user: { email: "pat@company.example", emailVerified: false },
            source,
          },
          untouchable,
        ),
      ).toBeUndefined();
    }
  });
});
