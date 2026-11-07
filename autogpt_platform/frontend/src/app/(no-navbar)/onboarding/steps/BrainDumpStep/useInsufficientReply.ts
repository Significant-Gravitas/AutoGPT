import { trackBrainDump } from "@/services/onboarding/brain-dump-analytics";
import { useState } from "react";
import {
  followUpPlaceholder,
  heardTextOf,
  type InsufficientMode,
} from "./components/insufficient-copy";
import { offYourPlateExampleForRole } from "./prompts";

interface ShowArgs {
  mode: InsufficientMode;
  errorCode: string;
  // The voice take's `transcript_preview`, or the text that was typed.
  heardText: string | null | undefined;
  // What the gate's judge wrote back to it (`otto_reply`), when it did.
  ottoReply?: string | null;
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
  const [ottoReply, setOttoReply] = useState<string | null>(null);
  const [isFollowUp, setIsFollowUp] = useState(false);

  function show(args: ShowArgs) {
    const heard = heardTextOf(args.heardText);
    const written = heard ? heardTextOf(args.ottoReply) : null;
    setMode(args.mode);
    setHeardText(heard);
    setOttoReply(written);
    trackBrainDump("brain_dump_insufficient_shown", {
      mode: args.mode,
      error_code: args.errorCode,
      heard: heard !== null,
      ai_reply: written !== null,
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
    ottoReply,
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
