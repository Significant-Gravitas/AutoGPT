import { describe, expect, it } from "vitest";
import { isDiffText, parseUnifiedDiff } from "../fileDiffHelpers";

describe("isDiffText", () => {
  it("recognizes unified diffs with hunk metadata", () => {
    expect(isDiffText("@@ -1,2 +1,2 @@\n-old\n+new")).toBe(true);
  });

  it("does not classify ordinary plus/minus prose as a diff", () => {
    expect(isDiffText("- downside\n+ upside")).toBe(false);
  });
});

describe("parseUnifiedDiff", () => {
  it("skips file headers and newline markers and tracks line numbers", () => {
    const parsed = parseUnifiedDiff(
      "--- a/file.ts\n+++ b/file.ts\n@@ -3,2 +3,2 @@\n-old\n+new\n same\n\\ No newline at end of file",
    );

    expect(parsed).toEqual({
      rows: [
        { old: 3, cur: null, type: "del", text: "old" },
        { old: null, cur: 3, type: "add", text: "new" },
        { old: 4, cur: 4, type: "ctx", text: "same" },
      ],
      added: 1,
      removed: 1,
      truncated: false,
    });
  });

  it("caps very large previews while preserving totals", () => {
    const lines = Array.from({ length: 550 }, (_, index) => `+line ${index}`);
    const parsed = parseUnifiedDiff(`@@ -0,0 +1,550 @@\n${lines.join("\n")}`);

    expect(parsed.rows).toHaveLength(500);
    expect(parsed.added).toBe(550);
    expect(parsed.truncated).toBe(true);
  });
});
