import { describe, expect, it } from "vitest";
import {
  getExpertChatHref,
  getPresenceColor,
  getPresenceLabel,
} from "./helpers";

describe("getPresenceColor", () => {
  it("shows working experts in amber", () => {
    expect(getPresenceColor("working")).toBe("bg-amber-500");
  });

  it("greys out paused and needs-setup experts", () => {
    expect(getPresenceColor("paused")).toBe("bg-zinc-300");
    expect(getPresenceColor("needs_setup")).toBe("bg-zinc-300");
  });

  it("shows every other status as a live green dot", () => {
    expect(getPresenceColor("ready")).toBe("bg-emerald-500");
    expect(getPresenceColor("failed")).toBe("bg-emerald-500");
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
});

describe("getExpertChatHref", () => {
  it("deep-links into copilot by expert id", () => {
    expect(getExpertChatHref("expert-123")).toBe(
      "/copilot?expertId=expert-123",
    );
  });
});
