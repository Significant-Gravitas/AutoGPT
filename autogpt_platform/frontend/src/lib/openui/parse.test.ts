import { describe, expect, it } from "vitest";
import { parseWorkspace, workspaceText } from "./parse";
import {
  campaign,
  campaignPlan,
  failures,
  leads,
  outreach,
  performance,
} from "./samples";

describe("OpenUI workspace contracts", () => {
  it.each([
    performance,
    failures,
    leads,
    outreach,
    campaign,
    campaignPlan({ audience: 'A "quoted" audience\nwith newlines' }),
  ])("accepts a complete sample", (source) => {
    expect(parseWorkspace(source).typeName).toBe("Workspace");
  });
  it.each([
    'root = Workspace("Incomplete", "Missing section", [missing])',
    'root = Workspace("Truncated", "Test", [',
    'root = UnregisteredComponent("No")',
    'root = Workspace("Tools", "Not allowed", [])\nquery = Query("private_data", {})',
  ])("rejects incomplete or unsupported output", (source) => {
    expect(() => parseWorkspace(source)).toThrow();
  });
  it("retains numeric chart values in the text comparison", () => {
    expect(workspaceText(performance)).toContain("Mon: 126 runs");
    expect(workspaceText(performance)).toContain("Completed runs: 1,284");
  });
});
