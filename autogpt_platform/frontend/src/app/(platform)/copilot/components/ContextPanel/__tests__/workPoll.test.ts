import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import { describe, expect, it } from "vitest";
import {
  delegationsOf,
  WORK_POLL_CAP_MS,
  WORK_POLL_MS,
  workPollInterval,
} from "../workPoll";

function chat(output: Record<string, unknown>, activeStream = false) {
  return {
    id: "chat-1",
    created_at: "2026-09-28T00:00:00Z",
    updated_at: "2026-09-28T00:00:00Z",
    user_id: "u-1",
    active_stream: activeStream ? { turn_id: "t" } : null,
    messages: [
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            id: "call-1",
            function: {
              name: "delegate_to_expert",
              arguments: JSON.stringify({ expert_id: "exp-alex" }),
            },
          },
        ],
      },
      {
        role: "tool",
        tool_call_id: "call-1",
        content: JSON.stringify(output),
      },
    ],
  } as unknown as SessionDetailResponse;
}

const NOW = 10_000_000;
const RUNNING = chat({ status: "running", sub_session_id: "sub-1" });

describe("workPollInterval", () => {
  it("polls while a teammate's own session says they are live", () => {
    expect(
      workPollInterval({
        session: RUNNING,
        liveStatuses: { "call-1": "running" },
        armedAt: NOW,
        now: NOW,
      }),
    ).toBe(WORK_POLL_MS);
  });

  it("stops once the live status is done, whatever the transcript froze", () => {
    expect(
      workPollInterval({
        session: RUNNING,
        liveStatuses: { "call-1": "completed" },
        armedAt: NOW,
        now: NOW,
      }),
    ).toBe(false);
  });

  it("does not poll on a frozen transcript status alone", () => {
    expect(
      workPollInterval({
        session: RUNNING,
        liveStatuses: {},
        armedAt: NOW,
        now: NOW,
      }),
    ).toBe(false);
  });

  it("stops after the cap, even while live or streaming", () => {
    const later = NOW + WORK_POLL_CAP_MS + 1;
    expect(
      workPollInterval({
        session: RUNNING,
        liveStatuses: { "call-1": "needs-input" },
        armedAt: NOW,
        now: later,
      }),
    ).toBe(false);
    expect(
      workPollInterval({
        session: chat({ status: "running" }, true),
        liveStatuses: {},
        armedAt: NOW,
        now: later,
      }),
    ).toBe(false);
  });

  it("polls while the chat's own turn streams", () => {
    expect(
      workPollInterval({
        session: chat({ status: "running" }, true),
        liveStatuses: {},
        armedAt: NOW,
        now: NOW,
      }),
    ).toBe(WORK_POLL_MS);
  });
});

describe("delegationsOf", () => {
  it("converts a fetched session once", () => {
    const first = delegationsOf(RUNNING);
    expect(first).toHaveLength(1);
    expect(delegationsOf(RUNNING)).toBe(first);
  });
});
