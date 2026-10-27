"use client";

import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import {
  GREETING_ORB_LAYOUT_ID,
  ORB_FLIP_TRANSITION,
  ORB_FLIP_TRANSITION_REDUCED,
} from "../../helpers/greetingOrb";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { useToast } from "@/components/molecules/Toast/use-toast";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipPortal,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { Copy01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { motion, useReducedMotion } from "framer-motion";
import { useState } from "react";
import { SuggestedTaskCards } from "../SuggestedTasks/SuggestedTaskCards";
import type { useSuggestedTasks } from "../SuggestedTasks/useSuggestedTasks";

interface Props {
  name: string | null;
  greeting: string;
  suggestions: ReturnType<typeof useSuggestedTasks>;
  transcript?: string;
  disabled?: boolean;
}

// Otto's avatar travels into this card's heading from the loader, so the
// heading is revealed on its arrival and everything below waits for the
// trip to finish.
const HEADING_START = 0.2;
const GREETING_START = 0.5;
const GREETING_DURATION = 0.45;
const ROW_START_BUFFER = 0.3;
const FOOTER_BUFFER = 0.35;

// One reveal schedule shared with EmptySession so the composer can enter
// after everything here has finished. The greeting lands in one piece, the
// prompt list follows as one block, then the footer, then the composer.
export function introRevealTimings() {
  const promptsStart = GREETING_START + GREETING_DURATION + ROW_START_BUFFER;
  const footerStart = promptsStart + FOOTER_BUFFER;
  const composerStart = footerStart + 0.4;
  return { promptsStart, footerStart, composerStart };
}

// The first thing a user sees after onboarding. Replaces the regular
// empty-session hero entirely: heading first, then the greeting word by
// word, then the prompt rows one after another, footer line last.
export function OnboardingIntroCard({
  name,
  greeting,
  suggestions,
  transcript = "",
  disabled = false,
}: Props) {
  const prefersReducedMotion = useReducedMotion();
  const [isCopied, setIsCopied] = useState(false);
  const { toast } = useToast();

  async function handleCopyTranscript() {
    // Denied clipboard permission, or a non-secure origin. Showing the
    // tick regardless would claim a copy that never happened.
    try {
      await navigator.clipboard.writeText(transcript);
    } catch {
      toast({
        title: "Could not copy the transcript",
        description: "Your browser blocked clipboard access.",
        variant: "destructive",
      });
      return;
    }
    setIsCopied(true);
    setTimeout(() => setIsCopied(false), 2000);
  }
  const { promptsStart, footerStart } = introRevealTimings();

  function reveal(delay: number) {
    if (prefersReducedMotion) {
      return {
        initial: { opacity: 0 },
        animate: { opacity: 1 },
        transition: { duration: 0.3, delay },
      };
    }
    return {
      initial: { opacity: 0, y: 10, filter: "blur(4px)" },
      animate: { opacity: 1, y: 0, filter: "blur(0px)" },
      transition: { duration: 0.45, ease: [0, 0, 0.2, 1] as const, delay },
    };
  }

  return (
    <div
      className="mb-8 w-full max-w-[48rem] text-left"
      data-testid="onboarding-intro-card"
    >
      <div className="mb-4 flex items-center gap-3">
        {/* Not revealed — it flies in from the loader's centre under its
            own layout animation. Fading it too would fight that trip. */}
        <motion.span
          layoutId={GREETING_ORB_LAYOUT_ID}
          transition={
            prefersReducedMotion
              ? ORB_FLIP_TRANSITION_REDUCED
              : ORB_FLIP_TRANSITION
          }
          className="relative block size-10 shrink-0"
        >
          <AutopilotAvatar size={40} transparent className="size-full" />
        </motion.span>
        <motion.div {...reveal(HEADING_START)}>
          <Text variant="large-medium" tone="primary">
            Hey
            {name ? (
              <>
                , <span className="text-foreground-strong">{name}</span>
              </>
            ) : (
              " there"
            )}
          </Text>
        </motion.div>
        {transcript && (
          <motion.div className="ml-auto" {...reveal(HEADING_START)}>
            <Tooltip inheritDelay>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  onClick={handleCopyTranscript}
                  aria-label="Copy your recording's transcript"
                  className="rounded-full p-2 text-foreground-subtle transition-colors hover:bg-surface-hover hover:text-foreground-body"
                >
                  {isCopied ? (
                    <Icon
                      icon={Tick02Icon}
                      size={16}
                      // TODO(tokens): no exact role for text-emerald-600
                      className="text-emerald-600"
                    />
                  ) : (
                    <Icon icon={Copy01Icon} size={16} />
                  )}
                </button>
              </TooltipTrigger>
              <TooltipPortal>
                <TooltipContent>
                  {isCopied ? "Copied!" : "Copy everything you told me"}
                </TooltipContent>
              </TooltipPortal>
            </Tooltip>
          </motion.div>
        )}
      </div>

      <motion.div {...reveal(GREETING_START)}>
        <Text variant="large" tone="secondary" className="text-pretty">
          {greeting}
        </Text>
      </motion.div>

      {suggestions.tasks.length > 0 && (
        <motion.div {...reveal(promptsStart)} className="mt-5">
          <SuggestedTaskCards
            tasks={suggestions.tasks}
            sendingId={suggestions.sendingId}
            disabled={disabled}
            onSend={suggestions.sendTask}
            onHide={suggestions.hideTask}
          />
        </motion.div>
      )}

      <motion.div {...reveal(footerStart)}>
        <Text variant="body" tone="muted" className="mt-5">
          {suggestions.tasks.length > 0
            ? "Want to do something else? Just write it in the textbox below."
            : "Tell me what you'd like to start with in the textbox below."}
        </Text>
      </motion.div>
    </div>
  );
}
