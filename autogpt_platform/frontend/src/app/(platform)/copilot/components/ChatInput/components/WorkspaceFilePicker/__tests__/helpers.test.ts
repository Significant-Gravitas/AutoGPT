import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { describe, expect, it } from "vitest";
import { applySelection, orderByList, type Selection } from "../helpers";

const FILES = makeFiles(6);

describe("applySelection", () => {
  it("toggles a plain click and moves the anchor to it", () => {
    const first = applySelection(empty(), FILES, 2);
    expect(names(first)).toEqual(["f-2"]);
    expect(first.anchor).toBe("id-2");

    const second = applySelection(first, FILES, 2);
    expect(names(second)).toEqual([]);
    expect(second.anchor).toBe("id-2");
  });

  it("selects the inclusive range forwards from the anchor", () => {
    const anchored = applySelection(empty(), FILES, 1);

    const ranged = applySelection(anchored, FILES, 4, { shift: true });

    expect(names(ranged)).toEqual(["f-1", "f-2", "f-3", "f-4"]);
    expect(ranged.anchor).toBe("id-1");
  });

  it("selects the inclusive range backwards from the anchor", () => {
    const anchored = applySelection(empty(), FILES, 4);

    const ranged = applySelection(anchored, FILES, 1, { shift: true });

    expect(names(ranged)).toEqual(["f-1", "f-2", "f-3", "f-4"]);
    expect(ranged.anchor).toBe("id-4");
  });

  it("treats Shift with no anchor as a plain click", () => {
    const result = applySelection(empty(), FILES, 3, { shift: true });

    expect(names(result)).toEqual(["f-3"]);
    expect(result.anchor).toBe("id-3");
  });

  it("adds to the selection rather than replacing it", () => {
    const withFar = applySelection(empty(), FILES, 5);
    const anchored = applySelection(withFar, FILES, 0);

    const ranged = applySelection(anchored, FILES, 2, { shift: true });

    expect(names(ranged)).toEqual(["f-0", "f-1", "f-2", "f-5"]);
  });

  it("never clears a file a second range passes back over", () => {
    const anchored = applySelection(empty(), FILES, 0);
    const wide = applySelection(anchored, FILES, 4, { shift: true });

    const narrow = applySelection(wide, FILES, 2, { shift: true });

    expect(names(narrow)).toEqual(["f-0", "f-1", "f-2", "f-3", "f-4"]);
  });

  it("toggles without moving the anchor for Ctrl/Cmd", () => {
    const anchored = applySelection(empty(), FILES, 1);

    const withExtra = applySelection(anchored, FILES, 4, { meta: true });
    expect(withExtra.anchor).toBe("id-1");

    // The anchor stayed put, so the range still runs from the plain click.
    const ranged = applySelection(withExtra, FILES, 3, { shift: true });
    expect(names(ranged)).toEqual(["f-1", "f-2", "f-3", "f-4"]);
  });

  it("ranges from the anchor file after the list shifts under it", () => {
    const anchored = applySelection(empty(), FILES, 1);
    const shifted = [{ ...FILES[0], id: "id-new", name: "new" }, ...FILES];

    // f-1 now sits at index 2; index 4 is f-3.
    const ranged = applySelection(anchored, shifted, 4, { shift: true });

    expect(names(ranged)).toEqual(["f-1", "f-2", "f-3"]);
  });

  it("treats Shift as a plain click once the anchor file is gone", () => {
    const anchored = applySelection(empty(), FILES, 1);
    const without = FILES.filter((f) => f.id !== "id-1");

    const result = applySelection(anchored, without, 3, { shift: true });

    expect(names(result)).toEqual(["f-1", "f-4"]);
    expect(result.anchor).toBe("id-4");
  });

  it("ignores an index no file sits at", () => {
    const anchored = applySelection(empty(), FILES, 1);

    expect(applySelection(anchored, FILES, 99)).toBe(anchored);
  });
});

describe("orderByList", () => {
  it("returns the selection in the order the list shows it", () => {
    const backwards = applySelection(
      applySelection(empty(), FILES, 3),
      FILES,
      1,
      { shift: true },
    );

    expect(orderByList(backwards.selected, FILES).map((f) => f.name)).toEqual([
      "f-1",
      "f-2",
      "f-3",
    ]);
  });

  it("keeps a file the list no longer shows, at the end", () => {
    const selection = applySelection(empty(), FILES, 4);

    const ordered = orderByList(selection.selected, [FILES[0], FILES[1]]);

    expect(ordered.map((f) => f.name)).toEqual(["f-4"]);
  });
});

function empty(): Selection {
  return { selected: new Map(), anchor: null };
}

function names(selection: Selection) {
  return Array.from(selection.selected.values())
    .map((f) => f.name)
    .sort();
}

function makeFiles(count: number): WorkspaceFileItem[] {
  return Array.from({ length: count }, (_, i) => ({
    id: `id-${i}`,
    name: `f-${i}`,
    path: `/workspace/f-${i}`,
    mime_type: "text/plain",
    size_bytes: 10,
    origin: "uploaded",
    created_at: "2026-01-01T00:00:00Z",
  })) as WorkspaceFileItem[];
}
