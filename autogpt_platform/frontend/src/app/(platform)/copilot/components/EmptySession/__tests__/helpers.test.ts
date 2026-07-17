import type { User } from "@supabase/supabase-js";
import { describe, expect, it } from "vitest";
import { getGreetingName } from "../helpers";

function makeUser(metadata: Record<string, unknown>, email?: string): User {
  return { user_metadata: metadata, email } as unknown as User;
}

describe("getGreetingName", () => {
  it("prefers the name chosen during onboarding (preferred_name)", () => {
    const user = makeUser(
      { preferred_name: "Reinier", full_name: "R. van der Leer" },
      "reinier@example.com",
    );
    expect(getGreetingName(user)).toBe("Reinier");
  });

  it("ignores a blank preferred_name", () => {
    const user = makeUser({ preferred_name: "  ", full_name: "Jane Doe" });
    expect(getGreetingName(user)).toBe("Jane");
  });

  it("uses the first name from full_name when no preferred_name is set", () => {
    const user = makeUser({ full_name: "Jane Doe" });
    expect(getGreetingName(user)).toBe("Jane");
  });

  it("falls back to the name metadata field", () => {
    const user = makeUser({ name: "John Smith" });
    expect(getGreetingName(user)).toBe("John");
  });

  it("falls back to the email prefix when metadata has no name", () => {
    const user = makeUser({}, "jane.doe@example.com");
    expect(getGreetingName(user)).toBe("jane.doe");
  });

  it('falls back to "there" without a user', () => {
    expect(getGreetingName(null)).toBe("there");
    expect(getGreetingName(undefined)).toBe("there");
  });
});
