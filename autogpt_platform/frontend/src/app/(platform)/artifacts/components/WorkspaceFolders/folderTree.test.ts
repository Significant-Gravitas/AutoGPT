import { describe, expect, test } from "vitest";
import type { WorkspaceFolder } from "@/app/api/__generated__/models/workspaceFolder";
import {
  ancestorsOf,
  childrenOf,
  descendantIdsOf,
  folderSummary,
  subfolderCountOf,
} from "./folderTree";

function folder(
  id: string,
  name: string,
  parentId: string | null = null,
): WorkspaceFolder {
  return {
    id,
    workspace_id: "ws-1",
    name,
    parent_id: parentId,
    created_at: new Date("2026-01-01T00:00:00Z"),
    updated_at: new Date("2026-01-01T00:00:00Z"),
    file_count: 0,
  };
}

// Files
//   Reports
//     2026
//       Q3
//     Drafts
//   Archive
const TREE = [
  folder("reports", "Reports"),
  folder("archive", "Archive"),
  folder("y2026", "2026", "reports"),
  folder("drafts", "Drafts", "reports"),
  folder("q3", "Q3", "y2026"),
];

describe("childrenOf", () => {
  test("returns root folders sorted by name for a null parent", () => {
    expect(childrenOf(TREE, null).map((f) => f.name)).toEqual([
      "Archive",
      "Reports",
    ]);
  });

  test("returns only direct children of a folder", () => {
    expect(childrenOf(TREE, "reports").map((f) => f.name)).toEqual([
      "2026",
      "Drafts",
    ]);
  });

  test("treats a missing parent_id as root", () => {
    const legacy = [{ ...folder("a", "A"), parent_id: undefined }];
    expect(childrenOf(legacy, null).map((f) => f.id)).toEqual(["a"]);
  });

  test("returns nothing for a leaf", () => {
    expect(childrenOf(TREE, "q3")).toEqual([]);
  });
});

describe("ancestorsOf", () => {
  test("returns the chain root first, the folder itself last", () => {
    expect(ancestorsOf(TREE, "q3").map((f) => f.name)).toEqual([
      "Reports",
      "2026",
      "Q3",
    ]);
  });

  test("returns just the folder when it sits at the root", () => {
    expect(ancestorsOf(TREE, "reports").map((f) => f.name)).toEqual([
      "Reports",
    ]);
  });

  test("returns nothing for an unknown id", () => {
    expect(ancestorsOf(TREE, "gone")).toEqual([]);
  });

  test("stops at a parent the list does not contain", () => {
    const orphan = [folder("child", "Child", "missing-parent")];
    expect(ancestorsOf(orphan, "child").map((f) => f.name)).toEqual(["Child"]);
  });

  test("stops on a cycle instead of looping forever", () => {
    const cyclic = [folder("a", "A", "b"), folder("b", "B", "a")];
    expect(ancestorsOf(cyclic, "a").map((f) => f.name)).toEqual(["B", "A"]);
  });
});

describe("descendantIdsOf", () => {
  test("collects the whole subtree, excluding the folder itself", () => {
    expect([...descendantIdsOf(TREE, "reports")].sort()).toEqual([
      "drafts",
      "q3",
      "y2026",
    ]);
  });

  test("is empty for a leaf", () => {
    expect(descendantIdsOf(TREE, "q3").size).toBe(0);
  });

  test("terminates on a cycle", () => {
    const cyclic = [folder("a", "A", "b"), folder("b", "B", "a")];
    expect([...descendantIdsOf(cyclic, "a")].sort()).toEqual(["a", "b"]);
  });
});

describe("subfolderCountOf", () => {
  test("counts direct children only", () => {
    expect(subfolderCountOf(TREE, "reports")).toBe(2);
    expect(subfolderCountOf(TREE, "y2026")).toBe(1);
    expect(subfolderCountOf(TREE, "q3")).toBe(0);
  });
});

describe("folderSummary", () => {
  test("prefers the direct file count", () => {
    expect(folderSummary(3, 2)).toBe("3 files");
    expect(folderSummary(1, 0)).toBe("1 file");
  });

  test("falls back to subfolders when there are no direct files", () => {
    expect(folderSummary(0, 2)).toBe("2 folders");
    expect(folderSummary(0, 1)).toBe("1 folder");
  });

  test("reads Empty when the folder holds neither", () => {
    expect(folderSummary(0, 0)).toBe("Empty");
  });
});
