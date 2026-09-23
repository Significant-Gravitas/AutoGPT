import { beforeEach, describe, expect, it, vi } from "vitest";

const capture = vi.hoisted(() => vi.fn());
vi.mock("posthog-js", () => ({ default: { capture } }));

import { trackTabIntro } from "../tab-intro-analytics";

beforeEach(() => {
  capture.mockReset();
});

describe("trackTabIntro", () => {
  it("reports the event with its tab", () => {
    trackTabIntro("tab_intro_cta_clicked", {
      tab: "build",
      cta: "ask_autopilot",
    });

    expect(capture).toHaveBeenCalledWith("tab_intro_cta_clicked", {
      tab: "build",
      cta: "ask_autopilot",
    });
  });

  it("swallows a blocked analytics host rather than breaking a first visit", () => {
    capture.mockImplementation(() => {
      throw new Error("blocked by client");
    });

    expect(() =>
      trackTabIntro("tab_intro_shown", { tab: "agents" }),
    ).not.toThrow();
  });
});
