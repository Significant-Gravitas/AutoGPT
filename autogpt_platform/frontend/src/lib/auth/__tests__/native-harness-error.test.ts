import { describe, expect, it } from "vitest";
import { describeNativeAuthError } from "../native-harness-error";

describe("native auth harness diagnostics", () => {
  it("retains the actionable error while removing connection URLs and secrets", () => {
    const error = new Error(
      "ECONNREFUSED at postgresql://tester:private-password@127.0.0.1/db using temporary-signing-secret", // pragma: allowlist secret
    );
    const diagnostic = describeNativeAuthError(error, {
      BETTER_AUTH_SECRET: "temporary-signing-secret", // pragma: allowlist secret
    });
    expect(diagnostic).toContain("ECONNREFUSED");
    expect(diagnostic).not.toContain("private-password");
    expect(diagnostic).not.toContain("temporary-signing-secret");
    expect(diagnostic).not.toContain("postgresql://");
  });

  it("explains bind and module-load failures", () => {
    expect(
      describeNativeAuthError(new Error("listen EADDRINUSE 127.0.0.1"), {}),
    ).toContain("EADDRINUSE");
    expect(describeNativeAuthError("Cannot find module auth.ts", {})).toContain(
      "Cannot find module auth.ts",
    );
  });
});
