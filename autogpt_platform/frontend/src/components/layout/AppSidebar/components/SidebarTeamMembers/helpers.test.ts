import { describe, expect, it } from "vitest";
import type { HomeAgentStatusStatus } from "@/app/api/__generated__/models/homeAgentStatusStatus";
import { getExpertHref, getPresenceColor, getPresenceLabel } from "./helpers";

describe("getPresenceColor", () => {
  it("shows working experts in amber", () => {
    expect(getPresenceColor("working")).toBe("bg-amber-500");
  });

  it("greys out paused and needs-setup experts", () => {
    expect(getPresenceColor("paused")).toBe("bg-zinc-300");
    expect(getPresenceColor("needs_setup")).toBe("bg-zinc-300");
  });

  it("flags failed experts in red", () => {
    expect(getPresenceColor("failed")).toBe("bg-red-500");
  });

  it("shows ready experts as a live green dot", () => {
    expect(getPresenceColor("ready")).toBe("bg-emerald-500");
  });

  it("falls back safely for a newer backend status", () => {
    expect(getPresenceColor("future_status" as HomeAgentStatusStatus)).toBe(
      "bg-zinc-300",
    );
  });
});

describe("getPresenceLabel", () => {
  it("describes the actual status for assistive tech", () => {
    expect(getPresenceLabel("working")).toBe("Working");
    expect(getPresenceLabel("paused")).toBe("Paused");
    expect(getPresenceLabel("needs_setup")).toBe("Needs setup");
    expect(getPresenceLabel("failed")).toBe("Needs attention");
    expect(getPresenceLabel("ready")).toBe("Ready");
  });

  it("labels a newer backend status without crashing", () => {
    expect(getPresenceLabel("future_status" as HomeAgentStatusStatus)).toBe(
      "Unknown",
    );
  });
});

describe("getExpertHref", () => {
  it("links to the expert's detail page", () => {
    expect(getExpertHref("expert-123")).toBe("/team/expert-123");
  });

  it("url-encodes the expert id", () => {
    expect(getExpertHref("a/b&c")).toBe("/team/a%2Fb%26c");
  });
});
