import { afterEach, describe, expect, it, vi } from "vitest";
import {
  encouragementAt,
  formatElapsed,
  headline,
  isPermissionDenied,
  pickMimeType,
  recordingFeedbackAt,
  RING_TARGET_SECONDS,
  ringProgress,
} from "../helpers";

describe("headline", () => {
  it("greets by name and trims the stored value", () => {
    expect(headline("  Ada  ")).toBe("What keeps stealing your week, Ada?");
  });

  it("drops the name when the wizard never collected one", () => {
    expect(headline("   ")).toBe("What keeps stealing your week?");
  });
});

describe("encouragementAt", () => {
  it("shows nothing before the first line is due", () => {
    expect(encouragementAt(0)).toBeNull();
    expect(encouragementAt(19.9)).toBeNull();
  });

  it("shows a line for six seconds and then goes quiet", () => {
    expect(encouragementAt(20)).toBe("Keep going, this is gold");
    expect(encouragementAt(25.9)).toBe("Keep going, this is gold");
    expect(encouragementAt(26)).toBeNull();
  });

  it("uses twenty-second milestones through the second minute", () => {
    expect(encouragementAt(40)).not.toBeNull();
    expect(encouragementAt(60)).not.toBeNull();
    expect(encouragementAt(80)).not.toBeNull();
    expect(encouragementAt(100)).not.toBeNull();
    expect(encouragementAt(120)).not.toBeNull();
    expect(encouragementAt(126)).toBeNull();
  });

  it("uses thirty-second milestones after the second minute", () => {
    expect(encouragementAt(149.9)).toBeNull();
    expect(encouragementAt(150)).not.toBeNull();
    expect(encouragementAt(180)).not.toBeNull();
    expect(encouragementAt(240)).not.toBeNull();
    expect(encouragementAt(360)).not.toBeNull();
  });

  it("stays quiet after the six-minute message", () => {
    expect(encouragementAt(366)).toBeNull();
    expect(encouragementAt(600)).toBeNull();
  });
});

describe("recordingFeedbackAt", () => {
  it("shows duration guidance only after recording has settled in", () => {
    expect(recordingFeedbackAt(3.9)).toBeNull();
    expect(recordingFeedbackAt(4)).toBe("Most people talk for 2 to 3 minutes.");
    expect(recordingFeedbackAt(9.9)).toBe(
      "Most people talk for 2 to 3 minutes.",
    );
  });

  it("shows the first encouragement at twenty seconds", () => {
    expect(recordingFeedbackAt(10)).toBeNull();
    expect(recordingFeedbackAt(20)).toBe("Keep going, this is gold");
  });
});

describe("formatElapsed", () => {
  it("pads the seconds and floors the fraction", () => {
    expect(formatElapsed(0)).toBe("0:00");
    expect(formatElapsed(9.9)).toBe("0:09");
    expect(formatElapsed(95)).toBe("1:35");
    expect(formatElapsed(3600)).toBe("60:00");
  });
});

describe("ringProgress", () => {
  // A depth meter, not a limit: the ring holds at full and recording
  // carries on past it.
  it("fills toward the target and then holds", () => {
    expect(ringProgress(0)).toBe(0);
    expect(ringProgress(RING_TARGET_SECONDS / 2)).toBeCloseTo(0.5, 5);
    expect(ringProgress(RING_TARGET_SECONDS)).toBe(1);
    expect(ringProgress(RING_TARGET_SECONDS * 10)).toBe(1);
  });
});

describe("pickMimeType", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("falls back to webm where MediaRecorder does not exist", () => {
    expect(pickMimeType()).toBe("audio/webm");
  });

  it("uses webm when the browser supports it", () => {
    vi.stubGlobal("MediaRecorder", { isTypeSupported: () => true });
    expect(pickMimeType()).toBe("audio/webm");
  });

  // Safari only offers mp4, and the server allowlist has to match.
  it("falls back to mp4 when webm is unsupported", () => {
    vi.stubGlobal("MediaRecorder", {
      isTypeSupported: (type: string) => type !== "audio/webm",
    });
    expect(pickMimeType()).toBe("audio/mp4");
  });
});

describe("isPermissionDenied", () => {
  it("recognises the browser's refusal", () => {
    expect(isPermissionDenied(new DOMException("no", "NotAllowedError"))).toBe(
      true,
    );
  });

  // A missing microphone or a generic failure is not a denial — treating
  // it as one would send the user to the typed fallback for good.
  it("does not treat other failures as a denial", () => {
    expect(isPermissionDenied(new DOMException("gone", "NotFoundError"))).toBe(
      false,
    );
    expect(isPermissionDenied(new Error("NotAllowedError"))).toBe(false);
    expect(isPermissionDenied(null)).toBe(false);
  });
});
