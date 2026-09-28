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

  it("follows the polled session over the frozen transcript", () => {
    expect(resolve("running")).toBe("completed");
    expect(resolve("needs-input", { isLive: true })).toBe("running");
    expect(resolve("completed", { isLive: true })).toBe("completed");
    expect(resolve("running", { isError: true })).toBe("unknown");
    expect(resolve("cancelled")).toBe("cancelled");
    expect(resolve("running", { session: null })).toBe("running");
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
