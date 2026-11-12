import {
  patchV2UpdateFeedbackPrompt,
  postV2ClaimFeedbackPrompt,
} from "@/app/api/__generated__/endpoints/feedback-prompts/feedback-prompts";
import type { ClaimFeedbackPromptRequestKind } from "@/app/api/__generated__/models/claimFeedbackPromptRequestKind";
import type { UpdateFeedbackPromptRequest } from "@/app/api/__generated__/models/updateFeedbackPromptRequest";
import { useRef } from "react";

export type FeedbackPromptKind = ClaimFeedbackPromptRequestKind;
export type FeedbackPromptAnswer = UpdateFeedbackPromptRequest;

export interface FeedbackPromptClaim {
  id: string | null;
  /** No answer came back, as opposed to the backend refusing a showing. */
  failed: boolean;
}

async function requestClaim(
  kind: FeedbackPromptKind,
): Promise<FeedbackPromptClaim> {
  try {
    const response = await postV2ClaimFeedbackPrompt({ kind });
    if (response.status !== 200) return { id: null, failed: true };
    return { id: response.data.id ?? null, failed: false };
  } catch {
    return { id: null, failed: true };
  }
}

async function sendAnswer(
  claimed: Promise<FeedbackPromptClaim>,
  patch: FeedbackPromptAnswer,
) {
  try {
    const { id } = await claimed;
    if (id) await patchV2UpdateFeedbackPrompt(id, patch);
  } catch {
    // A prompt must never get in the way of what the person came to do.
  }
}

// One showing of a prompt. A claim with no id means "don't show it", and
// answers are written one after another so the last tap is the one that stands.
export function useFeedbackPrompt(kind: FeedbackPromptKind) {
  const claimRef = useRef<Promise<FeedbackPromptClaim> | null>(null);
  const savesRef = useRef<Promise<unknown>>(Promise.resolve());

  function claim(fresh = false) {
    if (fresh || !claimRef.current) claimRef.current = requestClaim(kind);
    return claimRef.current;
  }

  function adopt(id: string) {
    claimRef.current = Promise.resolve({ id, failed: false });
  }

  function save(patch: FeedbackPromptAnswer) {
    const claimed = claimRef.current;
    if (!claimed) return;
    savesRef.current = savesRef.current.then(() => sendAnswer(claimed, patch));
  }

  return { claim, adopt, save };
}
