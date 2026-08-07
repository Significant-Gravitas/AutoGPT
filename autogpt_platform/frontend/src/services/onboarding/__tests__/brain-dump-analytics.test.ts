import { beforeEach, describe, expect, it, vi } from "vitest";

const capture = vi.hoisted(() => vi.fn());
vi.mock("posthog-js", () => ({ default: { capture } }));

import { trackBrainDump } from "../brain-dump-analytics";

beforeEach(() => {
  capture.mockReset();
});

describe("trackBrainDump", () => {
  it("forwards the event name and properties to posthog", () => {
    trackBrainDump("Brain Dump Completed", { duration_secs: 42 });

    expect(capture).toHaveBeenCalledWith("Brain Dump Completed", {
      duration_secs: 42,
    });
  });

  it("passes undefined properties through rather than inventing an object", () => {
    trackBrainDump("Brain Dump Started");

    expect(capture).toHaveBeenCalledWith("Brain Dump Started", undefined);
  });

  it("swallows a throwing analytics host so a recording is never interrupted", () => {
    capture.mockImplementation(() => {
      throw new Error("posthog host blocked");
    });

    expect(() => trackBrainDump("Brain Dump Permission Denied")).not.toThrow();
  });

  it("keeps capturing after a failed capture", () => {
    capture.mockImplementationOnce(() => {
      throw new Error("posthog host blocked");
    });

    trackBrainDump("Brain Dump Retry");
    trackBrainDump("Intro Followup Sent", { source: "suggestion" });

    expect(capture).toHaveBeenCalledTimes(2);
    expect(capture).toHaveBeenLastCalledWith("Intro Followup Sent", {
      source: "suggestion",
    });
  });
});
