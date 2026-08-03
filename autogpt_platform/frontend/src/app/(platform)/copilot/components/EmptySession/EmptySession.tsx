"use client";

import { ChatInput } from "@/app/(platform)/copilot/components/ChatInput/ChatInput";
import { useGetV2GetSuggestedPrompts } from "@/app/api/__generated__/endpoints/chat/chat";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { DotDistortionShader } from "@/components/ui/dot-distortion-shader";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import {
  getGreetingName,
  getInputPlaceholder,
  getSuggestionThemes,
} from "./helpers";
import { SuggestionThemes } from "./components/SuggestionThemes/SuggestionThemes";
import { OnboardingIntroCard } from "../OnboardingIntroCard/OnboardingIntroCard";
import { OnboardingWelcomeDialog } from "../OnboardingWelcomeDialog/OnboardingWelcomeDialog";
import { useOnboardingIntroCard } from "../OnboardingIntroCard/useOnboardingIntroCard";
import { PulseChips } from "../PulseChips/PulseChips";
import { usePulseChips } from "../PulseChips/usePulseChips";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import type { WorkspaceAttachment } from "../../helpers/workspaceAttachments";
import { EmptyHero } from "./components/EmptyHero";
import { RecipientChip } from "../ChatInput/components/RecipientChip";
import { useRecipientPicker } from "./useRecipientPicker";

interface Props {
  inputLayoutId: string;
  isCreatingSession: boolean;
  onCreateSession: () => void | Promise<string>;
  onSend: (
    message: string,
    files?: File[],
    workspaceFiles?: WorkspaceAttachment[],
  ) => void | Promise<void>;
  isUploadingFiles?: boolean;
  droppedFiles?: File[];
  onDroppedFilesConsumed?: () => void;
  isAdoptingExpertSession?: boolean;
}

export function EmptySession({
  inputLayoutId,
  isCreatingSession,
  onSend,
  isUploadingFiles,
  droppedFiles,
  onDroppedFilesConsumed,
  isAdoptingExpertSession,
}: Props) {
  const { user } = useAuth();
  const greetingName = getGreetingName(user);
  const intro = useOnboardingIntroCard();
  const isBrainDumpEnabled = useGetFlag(Flag.ONBOARDING_BRAIN_DUMP);
  const isAgentBriefingEnabled = useGetFlag(Flag.AGENT_BRIEFING);
  const isExpertsEnabled = useGetFlag(Flag.HIRE_EXPERTS);
  const pulseChips = usePulseChips();
  const { options, recipient, isLoadingRecipient, selectRecipient } =
    useRecipientPicker();
  const isComposerDisabled = isCreatingSession || !!isAdoptingExpertSession;

  const { data: suggestedPromptsResponse, isLoading: isLoadingPrompts } =
    useGetV2GetSuggestedPrompts({
      query: { staleTime: Infinity, gcTime: Infinity, refetchOnMount: false },
    });
  const themes = getSuggestionThemes(
    suggestedPromptsResponse?.status === 200
      ? suggestedPromptsResponse.data.themes
      : undefined,
  );

  const [inputPlaceholder, setInputPlaceholder] = useState(
    getInputPlaceholder(),
  );

  useEffect(() => {
    function handleResize() {
      setInputPlaceholder(getInputPlaceholder(window.innerWidth));
    }
    handleResize();
    const mql = window.matchMedia("(max-width: 500px)");
    mql.addEventListener("change", handleResize);
    const mql2 = window.matchMedia("(max-width: 1080px)");
    mql2.addEventListener("change", handleResize);
    return () => {
      mql.removeEventListener("change", handleResize);
      mql2.removeEventListener("change", handleResize);
    };
  }, []);

  return (
    <div
      className={cn(
        "relative flex h-full flex-1 justify-center overflow-y-auto px-0 py-5 md:px-6 md:py-10",
        // The whole greeting flow reads top-down like a letter, so it
        // anchors to the top from its first visible frame; the regular
        // hero stays vertically centered.
        intro.anchorTop ? "items-start" : "items-center",
      )}
    >
      {!isBrainDumpEnabled && (
        <DotDistortionShader
          dotGap={14}
          dotSize={1}
          opacity={0.2}
          enableMouseInteraction={false}
          breathingSpeed={0.4}
          className="pointer-events-none absolute inset-0 !bg-transparent [&_canvas]:opacity-70"
        />
      )}
      <OnboardingWelcomeDialog
        isOpen={intro.isWelcomeOpen}
        onClose={intro.closeWelcome}
      />
      <motion.div
        className="relative z-10 w-full max-w-[52rem] text-center"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3 }}
      >
        <div className="mx-auto max-w-[52rem]">
          {intro.isVisible ? (
            <OnboardingIntroCard
              name={greetingName}
              greeting={intro.greeting}
              prompts={intro.prompts}
              transcript={intro.transcript}
              onSelectPrompt={onSend}
              disabled={isComposerDisabled}
            />
          ) : (
            // The regular hero also renders behind the welcome modal's
            // blur and while the greeting is still generating — it swaps
            // to the greeting the moment the real one arrives. Through
            // that whole flow it wears the greeting page's own layout so
            // the heading never moves when the swap happens.
            <EmptyHero
              name={greetingName}
              isAwaitingGreeting={intro.isAwaitingGreeting}
              isGreetingFlow={intro.anchorTop}
            />
          )}

          {isAgentBriefingEnabled &&
            !intro.isVisible &&
            !intro.isAwaitingGreeting && (
              <PulseChips chips={pulseChips} onChipClick={onSend} />
            )}

          {/* Held back while the greeting is on its way — it enters with
              the greeting page instead of sitting under a bare hero. */}
          {!intro.isAwaitingGreeting && (
            <div className={cn("mb-6", intro.isVisible && "max-w-[48rem]")}>
              <motion.div
                layoutId={inputLayoutId}
                transition={{ type: "spring", bounce: 0.2, duration: 0.65 }}
                className={cn(
                  isBrainDumpEnabled
                    ? "overflow-hidden rounded-xlarge border text-left transition-colors duration-300 ease-out"
                    : "w-full px-2",
                  // The greeting's prompt card bleeds 1.25rem past the text
                  // (-mx-5); the composer stretches the same amount so their
                  // borders line up. The regular hero keeps it centered.
                  isBrainDumpEnabled &&
                    (intro.isVisible
                      ? "-mx-5 max-w-[50.5rem]"
                      : "mx-auto w-full max-w-[42rem]"),
                )}
                style={
                  isBrainDumpEnabled
                    ? {
                        borderColor: "#e4e4e7",
                        boxShadow: "0 2px 8px rgba(0,0,0,0.04)",
                      }
                    : undefined
                }
              >
                <ChatInput
                  inputId="chat-input-empty"
                  onSend={onSend}
                  disabled={isComposerDisabled}
                  hideSubmitWhenEmpty={Boolean(isBrainDumpEnabled)}
                  isUploadingFiles={isUploadingFiles}
                  placeholder={inputPlaceholder}
                  className={
                    isBrainDumpEnabled
                      ? "w-full [&_textarea]:min-h-[4.5rem]"
                      : "w-full"
                  }
                  droppedFiles={droppedFiles}
                  onDroppedFilesConsumed={onDroppedFilesConsumed}
                  recipientPicker={
                    isExpertsEnabled ? (
                      <RecipientChip
                        recipient={recipient}
                        options={options}
                        isLoading={isLoadingRecipient}
                        onSelect={selectRecipient}
                      />
                    ) : undefined
                  }
                />
              </motion.div>
            </div>
          )}
        </div>

        {/* The greeting page is deliberately quiet: its own prompts are
            the suggestions, so the theme chips stay out of the way. Also
            held while the greeting decision is pending. */}
        {!intro.isVisible &&
          !intro.isAwaitingGreeting &&
          (isLoadingPrompts ? (
            <div className="flex flex-wrap items-center justify-center gap-3">
              {Array.from({ length: 4 }, (_, i) => (
                <Skeleton key={i} className="h-10 w-28 shrink-0 rounded-full" />
              ))}
            </div>
          ) : (
            <SuggestionThemes
              themes={themes}
              onSend={onSend}
              disabled={isComposerDisabled}
            />
          ))}
      </motion.div>
    </div>
  );
}
