import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  markExpertThreadSource,
  takeExpertThreadSource,
  trackExpertThreadCreated,
} from "../expert-thread-source";

const capture = vi.hoisted(() => vi.fn());
vi.mock("posthog-js", () => ({ default: { capture } }));

beforeEach(() => {
  capture.mockReset();
  takeExpertThreadSource();
});

describe("expert thread source", () => {
  it("hands the marked source to the next thread only", () => {
    markExpertThreadSource("suggested_task");

    expect(takeExpertThreadSource()).toBe("suggested_task");
    expect(takeExpertThreadSource()).toBeNull();
  });

  it("has no source until one is marked", () => {
    expect(takeExpertThreadSource()).toBeNull();
  });

  it("drops a source that was cleared before a thread took it", () => {
    markExpertThreadSource("suggested_task");
    markExpertThreadSource(null);

    expect(takeExpertThreadSource()).toBeNull();
  });
});

describe("trackExpertThreadCreated", () => {
  it("reports where the thread was started from", () => {
    trackExpertThreadCreated("expert-maria", "suggested_task");

    expect(capture.mock.calls).toStrictEqual([
      [
        "expert_thread_created",
        { expert_id: "expert-maria", source: "suggested_task" },
      ],
    ]);
  });

  it("reports only the expert when the thread has no source", () => {
    trackExpertThreadCreated("expert-maria", null);

    expect(capture.mock.calls).toStrictEqual([
      ["expert_thread_created", { expert_id: "expert-maria" }],
    ]);
  });
});
