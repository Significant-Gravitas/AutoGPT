import { describe, expect, it } from "vitest";
import { getAnimationText } from "../helpers";

describe("FolderTool getAnimationText", () => {
  it("says workflows are moving while they move into a folder", () => {
    expect(
      getAnimationText({
        type: "tool-move_agents_to_folder",
        state: "input-available",
      }),
    ).toBe("Moving workflows…");
  });
});
