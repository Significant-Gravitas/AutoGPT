import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { describe, expect, it } from "vitest";
import {
  isTechnicalWorkspaceFile,
  workspaceFileOwner,
  workspaceFilePurpose,
  workspaceFileTitle,
  workspaceFileVerification,
} from "./workspace-file-visibility";

function file(overrides: Partial<WorkspaceFileItem> = {}): WorkspaceFileItem {
  return {
    id: "file-1",
    name: "phase0_positioning_icp.md",
    path: "/phase0_positioning_icp.md",
    mime_type: "text/markdown",
    size_bytes: 120,
    origin: "generated",
    created_at: "2026-08-28T00:00:00Z",
    metadata: {},
    ...overrides,
  };
}

describe("workspace file visibility", () => {
  it.each([
    { name: "agent.json" },
    { name: "build_state.json" },
    { name: "sdk-12345678-abcd.json" },
    { path: "/tool-outputs/search.json" },
    { metadata: { audience: "internal" } },
    { metadata: { artifact_role: "diagnostic" } },
  ])("recognizes technical output: $name$path", (overrides) => {
    expect(isTechnicalWorkspaceFile(file(overrides))).toBe(true);
  });

  it("keeps founder deliverables visible", () => {
    expect(isTechnicalWorkspaceFile(file())).toBe(false);
  });

  it("keeps a user-uploaded technical-looking filename visible", () => {
    expect(
      isTechnicalWorkspaceFile(
        file({ name: "agent.json", origin: "uploaded" }),
      ),
    ).toBe(false);
  });

  it("reads semantic founder metadata without exposing identifiers", () => {
    const deliverable = file({
      metadata: {
        title: "Positioning brief",
        owner_name: "Maya",
        purpose: "Choose the first customer segment",
        verification: "verified",
        work_item_id: "internal-work-id",
      },
    });

    expect(workspaceFileTitle(deliverable)).toBe("Positioning brief");
    expect(workspaceFileOwner(deliverable)).toBe("Maya");
    expect(workspaceFilePurpose(deliverable)).toBe(
      "Choose the first customer segment",
    );
    expect(workspaceFileVerification(deliverable)).toBe("Verified");
  });
});
