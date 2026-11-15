import { trackBrainDump } from "@/services/onboarding/brain-dump-analytics";
import { useState } from "react";
import {
  followUpPlaceholder,
  heardTextOf,
  type InsufficientMode,
  type OttoReplyKind,
} from "./components/insufficient-copy";
import { offYourPlateExampleForRole } from "./prompts";

interface ShowArgs {
  mode: InsufficientMode;
  errorCode: string;
  // The voice take's `transcript_preview`, or the text that was typed.
  heardText: string | null | undefined;
  // Which of Otto's fixed lines fits it (`otto_reply_kind`), when one was picked.
  replyKind?: OttoReplyKind | null;
}

// The not-enough screen's state: how the rejected take was sent, what Otto
// heard, and whether the person has since left the screen to talk or write
// again (a follow-up, which keeps Otto's question in view).
export function useInsufficientReply({
  role,
  promptSet,
}: {
  role: string;
  promptSet: string;
}) {
  const [mode, setMode] = useState<InsufficientMode>("voice");
  const [heardText, setHeardText] = useState<string | null>(null);
  const [replyKind, setReplyKind] = useState<OttoReplyKind | null>(null);
  const [isFollowUp, setIsFollowUp] = useState(false);

  function show(args: ShowArgs) {
    const heard = heardTextOf(args.heardText);
    const kind = heard ? (args.replyKind ?? null) : null;
    setMode(args.mode);
    setHeardText(heard);
    setReplyKind(kind);
    trackBrainDump("brain_dump_insufficient_shown", {
      mode: args.mode,
      error_code: args.errorCode,
      heard: heard !== null,
      reply_kind: kind,
      prompt_set: promptSet,
    });
  }

  function startFollowUp() {
    setIsFollowUp(true);
  }

  function endFollowUp() {
    setIsFollowUp(false);
  }

  // Added to an event only when true, so every event outside this flow
  // keeps exactly the properties it had.
  function followUpProperties() {
    return isFollowUp ? { follow_up: true } : {};
  }

  return {
    mode,
    heardText,
    replyKind,
    isFollowUp,
    typingPlaceholder: isFollowUp
      ? followUpPlaceholder(offYourPlateExampleForRole(role))
      : undefined,
    show,
    startFollowUp,
    endFollowUp,
    followUpProperties,
  };
}
