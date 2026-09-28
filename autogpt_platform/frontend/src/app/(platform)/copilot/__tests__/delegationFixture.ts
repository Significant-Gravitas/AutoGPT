import type { ChatDelegation } from "../delegations";

/** A hand-off as getChatDelegations builds it, for tests that need one. */
export function makeDelegation(
  overrides: Partial<ChatDelegation> = {},
): ChatDelegation {
  return {
    toolCallId: "c1",
    tool: "delegate_to_expert",
    expertId: "exp-alex",
    expert: null,
    prompt: "Draft the PRD",
    subSessionId: "sub-1",
    link: null,
    status: "running",
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
    ...overrides,
  };
}
