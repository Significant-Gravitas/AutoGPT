import { beforeEach, describe, expect, it, vi } from "vitest";

const capture = vi.hoisted(() => vi.fn());
vi.mock("posthog-js", () => ({ default: { capture } }));

import { trackBrainDump } from "../brain-dump-analytics";

beforeEach(() => {
  capture.mockReset();
});

describe("trackBrainDump", () => {
  it("forwards the event name and properties to posthog", () => {
    trackBrainDump("brain_dump_completed", { duration_secs: 42 });

    expect(capture).toHaveBeenCalledWith("brain_dump_completed", {
      duration_secs: 42,
    });
  });

  it("passes undefined properties through rather than inventing an object", () => {
    trackBrainDump("brain_dump_started");

    expect(capture).toHaveBeenCalledWith("brain_dump_started", undefined);
  });

  it("swallows a throwing analytics host so a recording is never interrupted", () => {
    capture.mockImplementation(() => {
      throw new Error("posthog host blocked");
    });

    expect(() => trackBrainDump("brain_dump_permission_denied")).not.toThrow();
  });

  it("keeps capturing after a failed capture", () => {
    capture.mockImplementationOnce(() => {
      throw new Error("posthog host blocked");
    });

    trackBrainDump("brain_dump_retry");
    trackBrainDump("intro_followup_sent", { source: "suggestion" });

    expect(capture).toHaveBeenCalledTimes(2);
    expect(capture).toHaveBeenLastCalledWith("intro_followup_sent", {
      source: "suggestion",
    });
  });
});
