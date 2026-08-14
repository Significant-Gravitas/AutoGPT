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
import { GreetingLoader } from "./components/GreetingLoader";
import { CopilotHome } from "../CopilotHome/CopilotHome";
import { NamingMomentCard } from "../NamingMomentCard/NamingMomentCard";
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
    <div className="relative flex h-full flex-1 items-start justify-center overflow-y-auto px-0 py-5 md:px-6 md:py-10">
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
        className={cn(
          "relative z-10 w-full max-w-[52rem] text-center",
          // The whole greeting flow reads top-down like a letter, so it
          // anchors to the top from its first visible frame; the regular
          // hero centers itself. `my-auto` rather than the parent's
          // `items-center`: auto margins collapse to 0 once the content is
          // taller than the scroller, where centering would push the top of
          // the page above the scroll origin and make it unreachable.
          !intro.anchorTop && "my-auto",
        )}
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
          ) : intro.isAwaitingGreeting ? (
            // Behind the welcome modal's blur and for as long as the
            // pipeline is still writing. The orb it renders is the same
            // element the card above puts in its heading, so the swap
            // moves it there rather than replacing it.
            <GreetingLoader />
          ) : (
            <EmptyHero name={greetingName} />
          )}

          {/* An existing user whose AI has never been named gets one nudge
              above the composer; the card gates itself on eligibility and
              renders nothing otherwise. */}
          {!intro.isVisible && !intro.isAwaitingGreeting && (
            <NamingMomentCard className="mb-4" />
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

          {/* The recap sits under the composer: the empty state's job is to
              get a message typed, so the briefing reads as context below it
              rather than as a wall above it. */}
          {!intro.isVisible &&
            !intro.isAwaitingGreeting &&
            (isExpertsEnabled ? (
              <div className="mx-auto mb-6 w-full max-w-[42rem]">
                <CopilotHome
                  fallback={
                    isAgentBriefingEnabled ? (
                      <PulseChips chips={pulseChips} onChipClick={onSend} />
                    ) : null
                  }
                />
              </div>
            ) : (
              isAgentBriefingEnabled && (
                <PulseChips chips={pulseChips} onChipClick={onSend} />
              )
            ))}
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
