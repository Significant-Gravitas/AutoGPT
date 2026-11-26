import { formatBudget, formatDollars, isExpertID, isProfileDirty, MODE_OPTIONS, parseBudget, validateProfile } from "../helpers";

test("budget: dollars in, credits out", () => {
  expect(parseBudget("25")).toEqual({ credits: 2500 });
  expect(parseBudget("12.50")).toEqual({ credits: 1250 });
  expect(parseBudget("0")).toEqual({ credits: 0 });
  expect(parseBudget("  ")).toEqual({ credits: null });
});
test.each(["12,50", "-5", "abc", "10001", "1.234"])("budget refuses %s", (text) => {
  expect(parseBudget(text)).toEqual({ error: "Enter a dollar amount like 25 or 12.50" });
});
test("budget shows as dollars", () => {
  expect(formatBudget(2500)).toBe("$25");
  expect(formatBudget(1250)).toBe("$12.50");
  expect(formatBudget(0)).toBe("No limit");
  expect(formatBudget(null)).toBe("Default");
});
test("profile needs a name and an identity within the server's limits", () => {
  const ok = { name: "Mia", identity: "Bookkeeper", voice_preferences: "", boundaries: "" };
  expect(validateProfile(ok)).toBeNull();
  expect(validateProfile({ ...ok, name: " " })).toBe("Give your expert a name.");
  expect(validateProfile({ ...ok, name: "x".repeat(101) })).toBe("Keep the name under 100 characters.");
  expect(validateProfile({ ...ok, identity: "" })).toBe("Say who this expert is.");
  expect(validateProfile({ ...ok, boundaries: "x".repeat(4001) })).toBe("Keep each section under 4,000 characters.");
  expect(isProfileDirty(ok, { ...ok, name: "Mia " })).toBe(false);
});
test("modes read as on the web", () => {
  expect(MODE_OPTIONS.map((o) => [o.value, o.label])).toEqual([["ask_first", "Ask first"], ["auto", "Auto"], ["unsupervised", "Unsupervised"]]);
});
test("spend shows plain dollars, even at zero", () => {
  expect(formatDollars(0)).toBe("$0");
  expect(formatDollars(123456)).toBe("$1,234.56");
});
test("only a UUID opens the manage screen", () => {
  expect(isExpertID("3f2b8c1e-4d5a-4b6c-8d7e-9f0a1b2c3d4e")).toBe(true);
  expect(isExpertID("../settings")).toBe(false);
  expect(isExpertID(undefined)).toBe(false);
});
