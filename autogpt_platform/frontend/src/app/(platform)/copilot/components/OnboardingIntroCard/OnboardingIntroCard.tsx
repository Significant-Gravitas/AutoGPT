"use client";

import { GlassOrb } from "@/components/molecules/GlassOrb/GlassOrb";
import type { SuggestedPrompt } from "@/app/api/__generated__/models/suggestedPrompt";
import {
  GREETING_ORB_LAYOUT_ID,
  ORB_FLIP_TRANSITION,
  ORB_FLIP_TRANSITION_REDUCED,
  SMALL_ORB_PARAMS,
} from "../../helpers/greetingOrb";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { useToast } from "@/components/molecules/Toast/use-toast";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Analytics01Icon,
  ArrowRight01Icon,
  CalendarCheckIcon,
  Chat01Icon,
  Clock01Icon,
  CodeIcon,
  Copy01Icon,
  DollarCircleIcon,
  File02Icon,
  FlashIcon,
  GlobeIcon,
  Mail01Icon,
  Megaphone01Icon,
  News01Icon,
  Notification02Icon,
  Robot01Icon,
  Rocket01Icon,
  Search01Icon,
  ShoppingCart01Icon,
  SparklesIcon,
  Target01Icon,
  Tick02Icon,
  UserGroupIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { motion, useReducedMotion } from "framer-motion";
import { useState } from "react";

// Slugs the backend may emit (see intro.PROMPT_ICONS); anything unknown
// falls back to the sparkle.
const PROMPT_ICONS: Record<string, IconSvgElement> = {
  sparkle: SparklesIcon,
  "chart-bar": Analytics01Icon,
  envelope: Mail01Icon,
  "magnifying-glass": Search01Icon,
  "calendar-check": CalendarCheckIcon,
  bell: Notification02Icon,
  "rocket-launch": Rocket01Icon,
  "file-text": File02Icon,
  globe: GlobeIcon,
  code: CodeIcon,
  newspaper: News01Icon,
  users: UserGroupIcon,
  "shopping-cart": ShoppingCart01Icon,
  chats: Chat01Icon,
  lightning: FlashIcon,
  target: Target01Icon,
  robot: Robot01Icon,
  clock: Clock01Icon,
  megaphone: Megaphone01Icon,
  "currency-dollar": DollarCircleIcon,
};

interface Props {
  name: string;
  greeting: string;
  prompts: SuggestedPrompt[];
  transcript?: string;
  onSelectPrompt: (prompt: string) => void;
  disabled?: boolean;
}

// The orb travels into this card's heading from the loader, so the
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
  prompts,
  transcript = "",
  onSelectPrompt,
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
          className="relative block size-8 shrink-0"
        >
          <GlassOrb params={SMALL_ORB_PARAMS} />
        </motion.span>
        <motion.div {...reveal(HEADING_START)}>
          <Text variant="large-medium" tone="primary">
            Hey, <span className="text-zinc-900">{name}</span>
          </Text>
        </motion.div>
        {transcript && (
          <motion.div className="ml-auto" {...reveal(HEADING_START)}>
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  onClick={handleCopyTranscript}
                  aria-label="Copy your recording's transcript"
                  className="rounded-full p-2 text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-zinc-700"
                >
                  {isCopied ? (
                    <Icon
                      icon={Tick02Icon}
                      size={16}
                      className="text-emerald-600"
                    />
                  ) : (
                    <Icon icon={Copy01Icon} size={16} />
                  )}
                </button>
              </TooltipTrigger>
              <TooltipContent>
                {isCopied ? "Copied!" : "Copy everything you told me"}
              </TooltipContent>
            </Tooltip>
          </motion.div>
        )}
      </div>

      <motion.div {...reveal(GREETING_START)}>
        <Text variant="large" tone="secondary" className="text-pretty">
          {greeting}
        </Text>
      </motion.div>

      {prompts.length > 0 && (
        <motion.ol
          {...reveal(promptsStart)}
          // Negative margins let the card's border breathe outward while
          // the row content (px-5) stays aligned with the text above it.
          className="-mx-5 mt-5 divide-y divide-zinc-100 overflow-hidden rounded-lg border border-zinc-200 bg-white"
        >
          {prompts.map((prompt) => {
            const promptIcon = PROMPT_ICONS[prompt.icon ?? ""] ?? SparklesIcon;
            return (
              <li key={prompt.title} data-testid="onboarding-intro-prompt">
                <button
                  type="button"
                  disabled={disabled}
                  // Sends the full prompt as the user's first message —
                  // which also creates the session and retires the greeting
                  // via the regular first-send path in useCopilotPage.
                  onClick={() => onSelectPrompt(prompt.prompt)}
                  className="group flex w-full cursor-pointer items-center gap-3 px-4 py-3 text-left transition-colors duration-150 hover:bg-zinc-50 disabled:cursor-default disabled:opacity-60 disabled:hover:bg-transparent"
                >
                  <Icon
                    icon={promptIcon}
                    size={15}
                    className="shrink-0 text-zinc-400"
                  />
                  <Text variant="body-medium" tone="primary">
                    {prompt.title}
                  </Text>
                  <Icon
                    icon={ArrowRight01Icon}
                    size={15}
                    className="ml-auto shrink-0 text-zinc-400 transition-transform duration-150 ease-out group-hover:translate-x-1"
                  />
                </button>
              </li>
            );
          })}
        </motion.ol>
      )}

      <motion.div {...reveal(footerStart)}>
        <Text variant="body" tone="muted" className="mt-5">
          Want to do something else? Just write it in the textbox below.
        </Text>
      </motion.div>
    </div>
  );
}
