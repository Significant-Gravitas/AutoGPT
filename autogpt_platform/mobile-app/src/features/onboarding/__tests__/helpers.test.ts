import { ROLES, canContinueRole, displayName, resolveRole } from "../helpers";

describe("ROLES", () => {
  it("sends the same ids the web sends", () => {
    expect(ROLES.map((role) => role.id)).toEqual([
      "Founder/CEO",
      "Operations",
      "Sales/BD",
      "Marketing",
      "Product/PM",
      "Engineering",
      "HR/People",
      "Other",
    ]);
  });
});

describe("canContinueRole", () => {
  it("needs a pick, and words for Other", () => {
    expect(canContinueRole("", "")).toBe(false);
    expect(canContinueRole("Sales/BD", "")).toBe(true);
    expect(canContinueRole("Other", "  ")).toBe(false);
    expect(canContinueRole("Other", "Studio owner")).toBe(true);
  });
});

describe("resolveRole", () => {
  it("is the pick, or what was written for Other", () => {
    expect(resolveRole("Marketing", "ignored")).toBe("Marketing");
    expect(resolveRole("Other", "  Studio owner ")).toBe("Studio owner");
  });

  it("cuts a long role to what the backend takes", () => {
    expect(Array.from(resolveRole("Other", "é".repeat(120)))).toHaveLength(100);
  });
});

describe("displayName", () => {
  it("is the first name, or the email's first part", () => {
    expect(displayName({ name: "Ada Lovelace", email: "ada@northwind.co" })).toBe(
      "Ada",
    );
    expect(displayName({ name: "", email: "ada@northwind.co" })).toBe("ada");
  });
});
