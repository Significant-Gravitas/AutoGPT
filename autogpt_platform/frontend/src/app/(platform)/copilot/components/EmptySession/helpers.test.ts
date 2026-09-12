import { describe, expect, test } from "vitest";
import { getExpertRoleLabel, getIntroLine } from "./helpers";

describe("getExpertRoleLabel", () => {
  // The nine roles the hire flow offers as presets (RoleStep/helpers.ts), which
  // is the shape most hired experts will carry.
  test.each([
    ["Marketer", "Marketer"],
    ["Developer", "Developer"],
    ["Researcher", "Researcher"],
    ["Writer", "Writer"],
    ["Analyst", "Analyst"],
    ["Recruiter", "Recruiter"],
    ["Sales", "Sales expert"],
    ["Support", "Support expert"],
    ["Operations", "Operations expert"],
  ])("labels the %s preset as %s", (role, expected) => {
    expect(getExpertRoleLabel(role)).toBe(expected);
  });

  test.each([
    ["Marketing Strategist", "Marketing Strategist"],
    ["Product Manager", "Product Manager"],
    ["Finance Director", "Finance Director"],
    ["Executive Assistant", "Executive Assistant"],
    ["Data Scientist", "Data Scientist"],
    ["Technician", "Technician"],
  ])("leaves the person-shaped custom role %s alone", (role, expected) => {
    expect(getExpertRoleLabel(role)).toBe(expected);
  });

  test.each([
    ["Marketing", "Marketing expert"],
    ["Customer Success", "Customer Success expert"],
    ["Social Media", "Social Media expert"],
    ["SEO", "SEO expert"],
    ["Legal", "Legal expert"],
  ])("calls the bare-domain custom role %s an expert", (role, expected) => {
    expect(getExpertRoleLabel(role)).toBe(expected);
  });

  // Only the head noun decides: "Customer" would pass the suffix test alone.
  test("judges a multi-word role by its last word", () => {
    expect(getExpertRoleLabel("Customer")).toBe("Customer");
    expect(getExpertRoleLabel("Customer Care")).toBe("Customer Care expert");
  });
});

describe("getIntroLine", () => {
  test("introduces an expert whose role names a person", () => {
    expect(getIntroLine({ name: "Maria", role: "Marketing Strategist" })).toBe(
      "I'm Maria, your Marketing Strategist. What should I take on?",
    );
  });

  test("introduces an expert whose role is a bare domain", () => {
    expect(getIntroLine({ name: "Sam", role: "Sales" })).toBe(
      "I'm Sam, your Sales expert. What should I take on?",
    );
  });

  test("drops the role clause when there is no role", () => {
    expect(getIntroLine({ name: "Max", role: null })).toBe(
      "I'm Max. What should I take on?",
    );
    expect(getIntroLine({ name: "Max", role: "   " })).toBe(
      "I'm Max. What should I take on?",
    );
  });

  test("falls back to the Otto line without an expert", () => {
    expect(getIntroLine(null)).toBe(
      "Tell me about your work — I'll find what to automate.",
    );
  });
});
