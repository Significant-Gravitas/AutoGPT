import { describe, expect, test } from "vitest";
import { signupFormSchema } from "../auth";

describe("signupFormSchema", () => {
  test("rejects invalid signup input", () => {
    const result = signupFormSchema.safeParse({
      email: "not-an-email",
      password: "short",
      confirmPassword: "different",
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
  });

  test("accepts a valid signup payload without a terms checkbox", () => {
    const result = signupFormSchema.safeParse({
      email: "valid@example.com",
      password: "validpassword123",
      confirmPassword: "validpassword123",
    });

    expect(result.success).toBe(true);
  });

  test("defaults to not opted out of marketing emails", () => {
    const result = signupFormSchema.parse({
      email: "valid@example.com",
      password: "validpassword123",
      confirmPassword: "validpassword123",
    });

    expect(result.marketingOptOut).toBe(false);
  });

  test("keeps a marketing opt-out", () => {
    const result = signupFormSchema.parse({
      email: "valid@example.com",
      password: "validpassword123",
      confirmPassword: "validpassword123",
      marketingOptOut: true,
    });

    expect(result.marketingOptOut).toBe(true);
  });

  test("rejects a non-boolean marketing opt-out", () => {
    const result = signupFormSchema.safeParse({
      email: "valid@example.com",
      password: "validpassword123",
      confirmPassword: "validpassword123",
      marketingOptOut: "yes",
    });

    expect(result.success).toBe(false);
  });
});
