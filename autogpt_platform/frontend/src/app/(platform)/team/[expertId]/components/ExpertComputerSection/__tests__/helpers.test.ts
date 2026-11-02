import type { ComputerInfo } from "@/app/api/__generated__/models/computerInfo";
import { describe, expect, it } from "vitest";
import { desktopActionLabel, formatScreen } from "../helpers";

function computer(
  box: "running" | "paused" | null,
  screenOn: boolean,
): ComputerInfo {
  return {
    owner_kind: "expert",
    owner_id: "expert-1",
    e2b_active: true,
    box: box
      ? {
          sandbox_id: "sb-1",
          state: box,
          started_at: new Date("2026-09-05T12:00:00Z"),
          cpu_count: 1,
          memory_mb: 2048,
          template_id: "agpt-desktop-1x2",
          mounts_attached: true,
        }
      : null,
    screen_on: screenOn,
    mounts: {},
    workspace_path: "/home/user/workspace",
    shared_path: "/home/user/shared",
  };
}

describe("desktopActionLabel", () => {
  it("offers to turn the screen on while it is off, box or no box", () => {
    expect(desktopActionLabel(null)).toBe("Turn on screen");
    expect(desktopActionLabel(computer(null, false))).toBe("Turn on screen");
    expect(desktopActionLabel(computer("running", false))).toBe(
      "Turn on screen",
    );
    expect(desktopActionLabel(computer("paused", false))).toBe(
      "Turn on screen",
    );
  });

  it("opens a running box's screen and resumes a suspended one's", () => {
    expect(desktopActionLabel(computer("running", true))).toBe("Open desktop");
    expect(desktopActionLabel(computer("paused", true))).toBe("Resume desktop");
  });
});

describe("formatScreen", () => {
  it("describes the screen for every box and screen state", () => {
    expect(formatScreen(computer(null, false))).toMatch(/comes on the first/);
    expect(formatScreen(computer("running", false))).toMatch(
      /inside this same machine/,
    );
    expect(formatScreen(computer("paused", true))).toMatch(/Screen on/);
  });
});
