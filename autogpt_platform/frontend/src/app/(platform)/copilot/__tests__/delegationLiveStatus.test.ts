import { describe, expect, it } from "vitest";
import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import { askedQuestionOf, resolveLiveStatus } from "../delegationLiveStatus";
import type { ChatDelegation, DelegationStatus } from "../delegations";

function delegation(status: DelegationStatus): ChatDelegation {
  return {
    toolCallId: "c1",
    tool: "delegate_to_expert",
    expertId: "exp-alex",
    expert: null,
    prompt: null,
    subSessionId: "sub-1",
    link: null,
    status,
    elapsedSeconds: null,
    costUsd: null,
    startedAt: null,
    finishedAt: null,
    response: null,
    question: null,
    questionOptions: [],
    error: null,
    files: [],
    reviewId: null,
    approved: false,
    superseded: false,
  };
}

const SESSION = { chat_status: "idle" } as SessionDetailResponse;
const NOW = 1_000_000;

function resolve(
  status: DelegationStatus,
  extra: Partial<Parameters<typeof resolveLiveStatus>[0]> = {},
) {
  return resolveLiveStatus({
    delegation: delegation(status),
    session: SESSION,
    isLive: false,
    question: null,
    isError: false,
    isPaused: false,
    answer: null,
    now: NOW,
    ...extra,
  });
}

describe("resolveLiveStatus", () => {
  it("needs the user whenever an idle teammate left a question", () => {
    expect(resolve("completed", { question: "Q4?" })).toBe("needs-input");
    expect(resolve("running", { question: "Q4?" })).toBe("needs-input");
  });

  it("reads a sent answer as the teammate resuming", () => {
    const answer = { question: "Q4?", text: "Q4", sentAt: NOW };
    expect(resolve("needs-input", { question: "Q4?", answer })).toBe("running");
    expect(resolve("needs-input", { answer })).toBe("running");
    expect(
      resolve("needs-input", { answer: { ...answer, sentAt: NOW - 60_000 } }),
    ).toBe("completed");
  });

  it("reads a thread stopped from the Work panel as stopped, not done", () => {
    function settled(...messages: Record<string, unknown>[]) {
      return resolve("running", {
        session: { chat_status: "idle", messages } as SessionDetailResponse,
      });
    }
    const brief = { role: "user", content: "Draft the PRD" };
    expect(
      settled(brief, {
        role: "assistant",
        content: "[__COPILOT_ERROR_f7a1__] Operation cancelled",
      }),
    ).toBe("cancelled");
    expect(
      settled(brief, {
        role: "assistant",
        content: "[__COPILOT_RETRYABLE_ERROR_a9c2__] The provider is busy",
      }),
    ).toBe("failed");
    // The turn died before writing a reply.
    expect(settled(brief)).toBe("failed");
    expect(
      settled(brief, {
        role: "assistant",
        content: "The PRD is in your files.",
      }),
    ).toBe("completed");
  });

  it("reads a hand-off that never started by the queue's note", () => {
    function settled(...messages: Record<string, unknown>[]) {
      return resolve("queued", {
        session: { chat_status: "idle", messages } as SessionDetailResponse,
      });
    }
    const brief = { role: "user", content: "Draft the PRD" };
    expect(
      settled(brief, {
        role: "assistant",
        content: "This task was cancelled before it started.",
        metadata: { queued_turn_refused: true },
      }),
    ).toBe("cancelled");
    expect(
      settled(brief, {
        role: "assistant",
        content: "Could not queue this task right now; try again shortly.",
        metadata: { queued_turn_refused: true },
      }),
    ).toBe("failed");
  });

  it("never reads a hand-off that landed on a limit as done", () => {
    function settled(
      pause: Record<string, unknown>,
      metadata: Record<string, unknown> = {},
    ) {
      return resolve("running", {
        session: {
          chat_status: "idle",
          metadata,
          messages: [
            { role: "user", content: "Draft the PRD" },
            {
              role: "assistant",
              content: "Paused.",
              metadata: { turn_paused: pause },
            },
          ],
        } as unknown as SessionDetailResponse,
      });
    }
    // Out of today's budget part-way: stopped, picked up from the thread.
    expect(settled({ reason: "daily", started: true })).toBe("cancelled");
    // Refused before it ran.
    expect(settled({ reason: "daily", started: false })).toBe("failed");
    // Stopped at its hand-off cap.
    expect(
      settled(
        { reason: "task", started: true, cap_usd: 2 },
        { delegation_cap_stopped: true },
      ),
    ).toBe("failed");
  });

  it("follows the polled session over the frozen transcript", () => {
    expect(resolve("running")).toBe("completed");
    expect(resolve("needs-input", { isLive: true })).toBe("running");
    // Running again (answered in their thread, cap raised): live wins.
    expect(resolve("completed", { isLive: true })).toBe("running");
    expect(
      resolve("completed", {
        isLive: true,
        session: { chat_status: "queued" } as SessionDetailResponse,
      }),
    ).toBe("queued");
    expect(resolve("running", { isError: true })).toBe("unknown");
    expect(resolve("cancelled")).toBe("cancelled");
    expect(resolve("running", { session: null })).toBe("running");
  });
});

describe("resolveLiveStatus for a superseded run", () => {
  it("keeps the older entry's result while a re-delegation uses the thread", () => {
    const older = { ...delegation("completed"), superseded: true };
    const inputs = {
      session: SESSION,
      question: null,
      isError: false,
      isPaused: false,
      answer: null,
      now: NOW,
    };
    expect(
      resolveLiveStatus({ ...inputs, delegation: older, isLive: true }),
    ).toBe("completed");
    expect(
      resolveLiveStatus({
        ...inputs,
        delegation: older,
        isLive: false,
        question: "Q4?",
      }),
    ).toBe("completed");
  });
});

describe("askedQuestionOf", () => {
  it("reads the last question and its option chips", () => {
    expect(
      askedQuestionOf([
        { name: "web_search", input: {} },
        {
          name: "ask_question",
          input: {
            questions: [
              { question: "Q4 or December?", options: ["Q4", "Dec", 1] },
            ],
          },
        },
      ]),
    ).toEqual({ text: "Q4 or December?", options: ["Q4", "Dec"] });
    expect(askedQuestionOf([{ name: "web_search", input: {} }])).toBeNull();
  });
});
