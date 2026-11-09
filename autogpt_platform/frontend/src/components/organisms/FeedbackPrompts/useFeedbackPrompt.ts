import {
  patchV2UpdateFeedbackPrompt,
  postV2ClaimFeedbackPrompt,
} from "@/app/api/__generated__/endpoints/feedback-prompts/feedback-prompts";
import type { ClaimFeedbackPromptRequestKind } from "@/app/api/__generated__/models/claimFeedbackPromptRequestKind";
import type { UpdateFeedbackPromptRequest } from "@/app/api/__generated__/models/updateFeedbackPromptRequest";
import { useRef } from "react";

export type FeedbackPromptKind = ClaimFeedbackPromptRequestKind;
export type FeedbackPromptAnswer = UpdateFeedbackPromptRequest;

async function requestClaim(kind: FeedbackPromptKind) {
  try {
    const response = await postV2ClaimFeedbackPrompt({ kind });
    return response.status === 200 ? (response.data.id ?? null) : null;
  } catch {
    return null;
  }
}

async function sendAnswer(
  claimed: Promise<string | null>,
  patch: FeedbackPromptAnswer,
) {
  try {
    const id = await claimed;
    if (id) await patchV2UpdateFeedbackPrompt(id, patch);
  } catch {
    // A prompt must never get in the way of what the person came to do.
  }
}

// One showing of a prompt. A claim that fails reads as "don't show it", and
// answers are written one after another so the last tap is the one that stands.
export function useFeedbackPrompt(kind: FeedbackPromptKind) {
  const claimRef = useRef<Promise<string | null> | null>(null);
  const savesRef = useRef<Promise<unknown>>(Promise.resolve());

  function claim() {
    claimRef.current ??= requestClaim(kind);
    return claimRef.current;
  }

  function adopt(id: string) {
    claimRef.current = Promise.resolve(id);
  }

  function save(patch: FeedbackPromptAnswer) {
    const claimed = claimRef.current;
    if (!claimed) return;
    savesRef.current = savesRef.current.then(() => sendAnswer(claimed, patch));
  }

  return { claim, adopt, save };
}
