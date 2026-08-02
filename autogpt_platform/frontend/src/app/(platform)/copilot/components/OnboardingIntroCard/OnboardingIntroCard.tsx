"use client";

import { GlassOrb } from "@/app/(no-navbar)/onboarding/steps/BrainDumpStep/components/GlassOrb/GlassOrb";
import type { GlassParams } from "@/app/(no-navbar)/onboarding/steps/BrainDumpStep/components/GlassOrb/GlassSurface";
import type { SuggestedPrompt } from "@/app/api/__generated__/models/suggestedPrompt";
import { Text } from "@/components/atoms/Text/Text";
import { TextGenerateEffect } from "@/components/ui/text-generate-effect";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  BellIcon,
  CalendarCheckIcon,
  CaretRightIcon,
  CheckIcon,
  CopyIcon,
  ChartBarIcon,
  ChatsIcon,
  ClockIcon,
  CodeIcon,
  CurrencyDollarIcon,
  EnvelopeIcon,
  FileTextIcon,
  GlobeIcon,
  type Icon,
  LightningIcon,
  MagnifyingGlassIcon,
  MegaphoneIcon,
  NewspaperIcon,
  RobotIcon,
  RocketLaunchIcon,
  ShoppingCartIcon,
  SparkleIcon,
  TargetIcon,
  UsersIcon,
} from "@phosphor-icons/react";
import { motion, useReducedMotion } from "framer-motion";
import { useState } from "react";

// Slugs the backend may emit (see intro.PROMPT_ICONS); anything unknown
// falls back to the sparkle.
const PROMPT_ICONS: Record<string, Icon> = {
  sparkle: SparkleIcon,
  "chart-bar": ChartBarIcon,
  envelope: EnvelopeIcon,
  "magnifying-glass": MagnifyingGlassIcon,
  "calendar-check": CalendarCheckIcon,
  bell: BellIcon,
  "rocket-launch": RocketLaunchIcon,
  "file-text": FileTextIcon,
  globe: GlobeIcon,
  code: CodeIcon,
  newspaper: NewspaperIcon,
  users: UsersIcon,
  "shopping-cart": ShoppingCartIcon,
  chats: ChatsIcon,
  lightning: LightningIcon,
  target: TargetIcon,
  robot: RobotIcon,
  clock: ClockIcon,
  megaphone: MegaphoneIcon,
  "currency-dollar": CurrencyDollarIcon,
};

interface Props {
  name: string;
  greeting: string;
  prompts: SuggestedPrompt[];
  transcript?: string;
  onSelectPrompt: (prompt: string) => void;
  disabled?: boolean;
}

// The default glass params are tuned for the big onboarding orb; at 32px
// that much frost and distortion collapses into a flat purple ball. Light
// frost + gentle refraction keeps the drifting blobs readable this small.
// Also rendered by EmptySession's hero while the greeting is on its way,
// so the orb is already on screen before the reveal.
export const SMALL_ORB_PARAMS: GlassParams = {
  frost: 1.5,
  saturation: 1.5,
  tint: 0.12,
  edge: 0.55,
  distortion: 8,
  ringWidth: 1,
  ringDepth: 2,
  ringDark: 0.25,
};

// The purple the orb's blobs blend into — the name mirrors it.
const ORB_PURPLE = "#8a4dff";

const GREETING_START = 0.35;
const WORD_STAGGER = 0.08;
const ROW_STAGGER = 0.12;
const ROW_START_BUFFER = 0.3;
const FOOTER_BUFFER = 0.35;

// One reveal schedule shared with EmptySession so the composer can enter
// after everything here has finished. All timings hang off the greeting's
// word count — the prompts wait for the last word, the footer waits for
// the last row, the composer comes after the footer.
export function introRevealTimings(greeting: string, promptCount: number) {
  const words = greeting.split(" ").filter(Boolean).length;
  const promptsStart = GREETING_START + words * WORD_STAGGER + ROW_START_BUFFER;
  const footerStart = promptsStart + promptCount * ROW_STAGGER + FOOTER_BUFFER;
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

  async function handleCopyTranscript() {
    await navigator.clipboard.writeText(transcript);
    setIsCopied(true);
    setTimeout(() => setIsCopied(false), 2000);
  }
  const { promptsStart, footerStart } = introRevealTimings(
    greeting,
    prompts.length,
  );

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
      <motion.div {...reveal(0)} className="mb-4 flex items-center gap-3">
        <span className="relative size-8 shrink-0">
          <GlassOrb params={SMALL_ORB_PARAMS} />
        </span>
        <Text variant="h3" className="!text-[1.25rem] text-zinc-800">
          Hey, <span style={{ color: ORB_PURPLE }}>{name}</span>
        </Text>
        {transcript && (
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                onClick={handleCopyTranscript}
                aria-label="Copy your recording's transcript"
                className="ml-auto rounded-full p-2 text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-zinc-700"
              >
                {isCopied ? (
                  <CheckIcon size={16} className="text-emerald-600" />
                ) : (
                  <CopyIcon size={16} />
                )}
              </button>
            </TooltipTrigger>
            <TooltipContent>
              {isCopied ? "Copied!" : "Copy everything you told me"}
            </TooltipContent>
          </Tooltip>
        )}
      </motion.div>

      <TextGenerateEffect
        words={greeting}
        duration={0.4}
        delay={GREETING_START}
        className="text-left !font-normal [&>div]:!mt-0 [&_div]:!text-[1.25rem] [&_div]:!leading-normal [&_div]:!tracking-normal [&_span]:!text-zinc-700"
      />

      {prompts.length > 0 && (
        <motion.ol
          {...reveal(promptsStart)}
          // Negative margins let the card's border breathe outward while
          // the row content (px-5) stays aligned with the text above it.
          className="-mx-5 mt-6 divide-y divide-zinc-100 overflow-hidden rounded-2xl border border-zinc-100 bg-white shadow-sm"
        >
          {prompts.map((prompt, index) => {
            const PromptIcon = PROMPT_ICONS[prompt.icon ?? ""] ?? SparkleIcon;
            return (
              <motion.li
                key={prompt.title}
                {...reveal(promptsStart + 0.15 + index * ROW_STAGGER)}
                data-testid="onboarding-intro-prompt"
              >
                <button
                  type="button"
                  disabled={disabled}
                  // Sends the full prompt as the user's first message —
                  // which also creates the session and retires the greeting
                  // via the regular first-send path in useCopilotPage.
                  onClick={() => onSelectPrompt(prompt.prompt)}
                  className="group flex w-full cursor-pointer items-center gap-4 px-5 py-4 text-left transition-colors duration-150 hover:bg-violet-50/60 disabled:cursor-default disabled:opacity-60 disabled:hover:bg-transparent"
                >
                  <PromptIcon
                    size={18}
                    className="shrink-0 text-violet-500"
                    weight="duotone"
                  />
                  <Text
                    variant="body-medium"
                    className="!text-[0.9375rem] !text-zinc-800"
                  >
                    {prompt.title}
                  </Text>
                  <CaretRightIcon
                    size={16}
                    className="ml-auto shrink-0 text-violet-500 transition-transform duration-150 ease-out group-hover:translate-x-1"
                  />
                </button>
              </motion.li>
            );
          })}
        </motion.ol>
      )}

      <motion.div {...reveal(footerStart)}>
        <Text variant="body" className="mt-6 !text-[0.9375rem] !text-zinc-500">
          Want to do something else? Just write it in the textbox below.
        </Text>
      </motion.div>
    </div>
  );
}
