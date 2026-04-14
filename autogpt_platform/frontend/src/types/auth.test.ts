import { describe, expect, test } from "vitest";
import { signupFormSchema } from "./auth";

describe("signupFormSchema", () => {
  test("rejects invalid signup input", () => {
    const result = signupFormSchema.safeParse({
      email: "not-an-email",
      password: "short",
      confirmPassword: "different",
      agreeToTerms: false,
    });

    expect(result.success).toBe(false);

    if (result.success) {
      return;
    }

    const { fieldErrors } = result.error.flatten();

    expect(fieldErrors.email?.length).toBeGreaterThan(0);
    expect(fieldErrors.password).toContain(
      "Password must contain at least 12 characters",
    );
    expect(fieldErrors.confirmPassword).toContain("Passwords don't match");
    expect(fieldErrors.agreeToTerms).toContain(
      "You must agree to the Terms of Use and Privacy Policy",
    );
  });

  test("accepts a valid signup payload", () => {
    const result = signupFormSchema.safeParse({
      email: "valid@example.com",
      password: "validpassword123",
      confirmPassword: "validpassword123",
      agreeToTerms: true,
    });

    expect(result.success).toBe(true);
  });
});
