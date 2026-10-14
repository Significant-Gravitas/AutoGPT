import { describe, expect, it } from "vitest";
import { isBoardPath, newProjectHref, nextProjectName } from "../helpers";

const B1 = "b1a2c3d4-5e6f-4a7b-8c9d-0e1f2a3b4c5d";

describe("nextProjectName", () => {
  it("starts with Untitled project", () => {
    expect(nextProjectName([])).toBe("Untitled project");
    expect(nextProjectName(["Home", "Launch"])).toBe("Untitled project");
  });

  it("numbers the next one from 2", () => {
    expect(nextProjectName(["Home", "Untitled project"])).toBe(
      "Untitled project 2",
    );
  });

  it("takes the first free number", () => {
    expect(
      nextProjectName([
        "Untitled project",
        "Untitled project 2",
        "Untitled project 4",
      ]),
    ).toBe("Untitled project 3");
  });
});

describe("newProjectHref", () => {
  it("opens the board with its name in edit mode", () => {
    expect(newProjectHref(B1)).toBe(`/boards/${B1}?rename=1`);
  });
});

describe("isBoardPath", () => {
  it("matches one board's page", () => {
    expect(isBoardPath(`/boards/${B1}`, B1)).toBe(true);
    expect(isBoardPath(`/boards/${B1}/x`, B1)).toBe(true);
    expect(isBoardPath(`/boards/${B1}0`, B1)).toBe(false);
    expect(isBoardPath("/home", B1)).toBe(false);
    expect(isBoardPath(null, B1)).toBe(false);
  });

  it("matches any board's page without an id", () => {
    expect(isBoardPath(`/boards/${B1}`)).toBe(true);
    expect(isBoardPath("/boardsx")).toBe(false);
    expect(isBoardPath("/home")).toBe(false);
  });
});
