import { ApiError } from "@/lib/autogpt-server-api/helpers";

export type ResetPhase = "confirm" | "busy" | "done";

// A reset waits for the expert's running work: a turn that ends mid-reset
// would put what it learned back into the memory the reset just erased.
export function isBusyError(error: unknown) {
  return error instanceof ApiError && error.status === 409;
}

export function resetFailedToast(expertName: string) {
  return {
    title: `Could not finish resetting ${expertName}`,
    description: "Please try again.",
    variant: "destructive" as const,
  };
}

// kickoff=1 has the expert open a new chat by introducing themselves, as
// after a hire. The reset removed every chat, so that is the next one.
export function getChatWithExpertHref(expertId: string) {
  return `/copilot?expertId=${encodeURIComponent(expertId)}&kickoff=1`;
}
