"use client";

import { ChatInput } from "@/app/(platform)/copilot/components/ChatInput/ChatInput";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import type { ReactNode } from "react";
import {
  getExpertInputPlaceholder,
  getGreetingName,
  getIntroLine,
} from "./helpers";
import { OnboardingIntroCard } from "../OnboardingIntroCard/OnboardingIntroCard";
import { OnboardingWelcomeDialog } from "../OnboardingWelcomeDialog/OnboardingWelcomeDialog";
import { useOnboardingIntroCard } from "../OnboardingIntroCard/useOnboardingIntroCard";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import type { WorkspaceAttachment } from "../../helpers/workspaceAttachments";
import { EmptyHero } from "./components/EmptyHero";
import { GreetingLoader } from "./components/GreetingLoader";
import { ExpertKickoffLoader } from "./components/ExpertKickoffLoader/ExpertKickoffLoader";
import { HomeRecap } from "@/app/(platform)/home/components/HomeRecap/HomeRecap";
import { RecipientChip } from "../ChatInput/components/RecipientChip";
import { ConnectionPicker } from "../ChatInput/components/ConnectionPicker/ConnectionPicker";
import { useRecipientPicker } from "./useRecipientPicker";
import { useHomeComposer } from "./useHomeComposer";

interface Props {
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
  isInteractionLocked?: boolean;
  isKickoffStarting?: boolean;
  expertName?: string;
  /** Expert the new conversation will address; scopes workspace-file pickers. */
  expertId?: string | null;
  /** Voice-mode toggle, rendered beside the mic. Absent when the flag is off. */
  voiceToggle?: ReactNode;
  /** The chat's approval-mode selector. Absent when the flag is off. */
  modeSelector?: ReactNode;
}

export function EmptySession({
  isCreatingSession,
  onSend,
  isUploadingFiles,
  droppedFiles,
  onDroppedFilesConsumed,
  isInteractionLocked,
  isKickoffStarting,
  expertName,
  expertId = null,
  voiceToggle,
  modeSelector,
}: Props) {
  const { user } = useAuth();
  const greetingName = getGreetingName(user);
  const intro = useOnboardingIntroCard();
  const isBrainDumpEnabled = useGetFlag(Flag.ONBOARDING_BRAIN_DUMP);
  const isExpertsEnabled = useGetFlag(Flag.HIRE_EXPERTS);
  const {
    options,
    recipient,
    selectedExpert,
    isLoadingRecipient,
    selectRecipient,
  } = useRecipientPicker();
  const isComposerDisabled = isCreatingSession || !!isInteractionLocked;
  const introLine = isLoadingRecipient ? null : getIntroLine(selectedExpert);
  const recipientPicker = isExpertsEnabled ? (
    <RecipientChip
      recipient={recipient}
      options={options}
      isLoading={isLoadingRecipient}
      onSelect={selectRecipient}
    />
  ) : undefined;

  const { inputPlaceholder } = useHomeComposer({
    enabled:
      isExpertsEnabled &&
      !intro.isVisible &&
      !intro.isAwaitingGreeting &&
      !isKickoffStarting,
  });

  if (isKickoffStarting) {
    return <ExpertKickoffLoader expertName={expertName} />;
  }

  return (
    <div className="relative flex h-full flex-1 items-start justify-center overflow-y-auto px-0 py-5 md:px-6 md:py-10">
      <OnboardingWelcomeDialog
        isOpen={intro.isWelcomeOpen}
        onClose={intro.closeWelcome}
      />
      {/* Which connection the new chat runs on, kept out of the composer and
          in the page corner, level with the inset header's controls. */}
      <div className="absolute right-3 top-3 z-30 empty:hidden">
        <ConnectionPicker className="ml-0" />
      </div>
      <motion.div
        className={cn(
          "relative z-10 w-full text-center",
          isExpertsEnabled ? "max-w-[1120px]" : "max-w-[52rem]",
          // The whole greeting flow reads top-down like a letter, so it
          // anchors to the top from its first visible frame; the regular
          // hero centers itself. `my-auto` rather than the parent's
          // `items-center`: auto margins collapse to 0 once the content is
          // taller than the scroller, where centering would push the top of
          // the page above the scroll origin and make it unreachable.
          !intro.anchorTop && !isExpertsEnabled && "my-auto",
        )}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3 }}
      >
        <div className="mx-auto max-w-[52rem] pt-6">
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
            <EmptyHero
              name={greetingName}
              intro={introLine}
              recipientPicker={recipientPicker}
              isExpert={Boolean(selectedExpert)}
              expertRole={selectedExpert?.role}
            />
          )}

          {/* Held back while the greeting is on its way — it enters with
              the greeting page instead of sitting under a bare hero. */}
          {!intro.isAwaitingGreeting && (
            <div className={cn("mb-6", intro.isVisible && "max-w-[48rem]")}>
              <div
                className={cn(
                  isBrainDumpEnabled
                    ? "text-left transition-colors duration-300 ease-out"
                    : "w-full px-2",
                  // The greeting's prompt card bleeds 1.25rem past the text
                  // (-mx-5); the composer stretches the same amount so their
                  // edges line up. No chrome of its own: the composer card is
                  // the only outline on screen. The regular hero keeps it
                  // centered.
                  isBrainDumpEnabled &&
                    (intro.isVisible
                      ? "-mx-5 max-w-[50.5rem]"
                      : "mx-auto w-full max-w-[42rem]"),
                )}
              >
                <ChatInput
                  inputId="chat-input-empty"
                  stacked
                  voiceToggle={voiceToggle}
                  modeSelector={modeSelector}
                  onSend={onSend}
                  disabled={isComposerDisabled}
                  isUploadingFiles={isUploadingFiles}
                  placeholder={
                    selectedExpert
                      ? getExpertInputPlaceholder(selectedExpert.name)
                      : inputPlaceholder
                  }
                  className={
                    isBrainDumpEnabled
                      ? "w-full [&_textarea]:min-h-[4.5rem]"
                      : "w-full"
                  }
                  droppedFiles={droppedFiles}
                  onDroppedFilesConsumed={onDroppedFilesConsumed}
                  expertId={expertId}
                  expertName={expertName}
                  recipientPicker={
                    intro.isVisible ? recipientPicker : undefined
                  }
                />
              </div>
            </div>
          )}
        </div>

        {!intro.isVisible && !intro.isAwaitingGreeting && isExpertsEnabled && (
          <HomeRecap />
        )}
      </motion.div>
    </div>
  );
}
