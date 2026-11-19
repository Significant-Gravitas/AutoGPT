import { expertRoleLabel } from "../expertRoleLabel";

describe("expertRoleLabel", () => {
  it("keeps a role that names a person", () => {
    expect(expertRoleLabel("Marketer")).toBe("Marketer");
  });

  it("adds expert to a domain role", () => {
    expect(expertRoleLabel("Support")).toBe("Support expert");
    expect(expertRoleLabel("Customer Success")).toBe("Customer Success expert");
  });

  it("does not read an inherited property as a preset role", () => {
    expect(expertRoleLabel("constructor")).toBe("constructor");
  });

  it("maps a preset role to its display label", () => {
    expect(expertRoleLabel("Sales")).toBe("Sales Development Rep");
  });
});
