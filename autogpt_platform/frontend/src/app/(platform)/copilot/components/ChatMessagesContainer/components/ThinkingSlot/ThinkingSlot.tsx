"use client";

import { useReducedMotion } from "framer-motion";
import {
  Message,
  MessageContent,
} from "@/components/organisms/Message/Message";
import { cn } from "@/lib/utils";
import type { TurnActivity } from "../../thinkingActivity";
import { ThinkingIndicator } from "../ThinkingIndicator";
import { useThinkingPresence } from "./useThinkingPresence";
import { useWantsThinking } from "./useWantsThinking";

interface Props {
  activity: TurnActivity;
  /** The turn is live and no other surface narrates the wait. */
  enabled: boolean;
  /** Another surface took over the wait (a restore, a live compaction): the
   *  indicator goes at once instead of fading. */
  handedOff: boolean;
  startedAtMs: number | null;
  /** The newest row is the assistant's, so the indicator reads as the end of
   *  that reply rather than a bubble of its own. */
  attached: boolean;
  compact: boolean;
}

/** The one place the thinking indicator renders for a live turn. It sits
 *  after the transcript for the whole turn, so the first streamed chunk never
 *  moves it into another subtree and remounts it. */
export function ThinkingSlot({
  activity,
  enabled,
  handedOff,
  startedAtMs,
  attached,
  compact,
}: Props) {
  const reducedMotion = useReducedMotion();
  const wanted = useWantsThinking(activity, enabled && !handedOff);
  const phase = useThinkingPresence(wanted, !!reducedMotion || handedOff);
  if (phase === "hidden") return null;

  return (
    <div
      className={cn(
        "transition-opacity duration-150 ease-out motion-reduce:transition-none",
        phase === "leaving" ? "opacity-0" : "opacity-100",
        attached && (compact ? "-mt-2" : "-mt-4"),
      )}
    >
      <Message from="assistant">
        <MessageContent>
          <div className="duration-200 ease-out animate-in fade-in slide-in-from-bottom-1 motion-reduce:animate-none">
            <ThinkingIndicator active startedAtMs={startedAtMs} />
          </div>
        </MessageContent>
      </Message>
    </div>
  );
}
