"use client";

import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { ChatInput } from "@/app/(platform)/copilot/components/ChatInput/ChatInput";
import { tourTarget } from "@/lib/tour-targets";
import { useFirstRunTourAutoStart } from "@/app/(platform)/components/FirstRunTour/useFirstRunTourAutoStart";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { useState, type ComponentProps, type ReactNode } from "react";
import type { WorkspaceAttachment } from "../../helpers/workspaceAttachments";
import { ConnectionPicker } from "../ChatInput/components/ConnectionPicker/ConnectionPicker";
import { RecipientChip } from "../ChatInput/components/RecipientChip";
import { OnboardingIntroCard } from "../OnboardingIntroCard/OnboardingIntroCard";
import { useOnboardingIntroCard } from "../OnboardingIntroCard/useOnboardingIntroCard";
import { ScaleLoader } from "../ScaleLoader/ScaleLoader";
import { SEND_AT_RESET_PLACEHOLDER } from "../SendAtReset/helpers";
import { useCollapsedSuggestions } from "../SuggestedTasks/useCollapsedSuggestions";
import { useSuggestedTasks } from "../SuggestedTasks/useSuggestedTasks";
import { ComposerTray } from "./components/ComposerTray/ComposerTray";
import { DitherTrail } from "./components/DitherTrail/DitherTrail";
import { EmptyHero } from "./components/EmptyHero";
import { ExpertInfoPanel } from "./components/ExpertInfoPanel/ExpertInfoPanel";
import { ExpertKickoffLoader } from "./components/ExpertKickoffLoader/ExpertKickoffLoader";
import { GreetingLoader } from "./components/GreetingLoader";
import { HomeBoard } from "./components/HomeBoard/HomeBoard";
import { HomeRecentRow } from "./components/HomeRecentChats/HomeRecentRow";
import { HomeScroller } from "./components/HomeScroller";
import { HomeViewTabs } from "./components/HomeViewTabs/HomeViewTabs";
import { useHomeView } from "./components/HomeViewTabs/useHomeView";
import { InfoToggle } from "./components/InfoToggle";
import {
  getExpertInputPlaceholder,
  getGreetingName,
  getIntroLine,
} from "./helpers";
import { useHomeComposer } from "./useHomeComposer";
import { useRecipientPicker } from "./useRecipientPicker";

interface Props {
  isCreatingSession: boolean;
  onCreateSession: () => void | Promise<string>;
  onSend: (
    message: string,
    files?: File[],
    workspaceFiles?: WorkspaceAttachment[],
  ) => void | Promise<void>;
  /** Sends a suggested task, resolving to the chat it started. */
  onStartTask?: (message: string) => Promise<string | undefined>;
  isUploadingFiles?: boolean;
  droppedFiles?: File[];
  onDroppedFilesConsumed?: () => void;
  isInteractionLocked?: boolean;
  isKickoffStarting?: boolean;
  expertName?: string;
  /** Expert the new conversation will address; scopes workspace-file pickers. */
  expertId?: string | null;
  /** Starts voice mode from the + menu. Absent when the flag is off. */
  voiceMode?: ComponentProps<typeof ChatInput>["voiceMode"];
  /** The chat's approval-mode selector. Absent when the flag is off. */
  modeSelector?: ReactNode;
  /** Out of AutoPilot budget with a reset ahead: the composer sets the new
   *  chat's first message to send at the reset. */
  atReset?: ComponentProps<typeof ChatInput>["atReset"];
  /** Shown above the composer while sending is paused. */
  usageNotice?: ReactNode;
  /** The running-low hint, shown under the composer. */
  lowHint?: ReactNode;
}

export function EmptySession({
  isCreatingSession,
  onSend,
  onStartTask,
  isUploadingFiles,
  droppedFiles,
  onDroppedFilesConsumed,
  isInteractionLocked,
  isKickoffStarting,
  expertName,
  expertId = null,
  voiceMode,
  modeSelector,
  atReset,
  usageNotice,
  lowHint,
}: Props) {
  const { user } = useAuth();
  const greetingName = getGreetingName(user);
  const intro = useOnboardingIntroCard();
  useFirstRunTourAutoStart({
    isPending: intro.isFirstRunPending,
    onFinished: intro.finishFirstRun,
  });

  const {
    options,
    experts,
    expertsById,
    isRosterLoaded,
    hasRosterFailed,
    recipient,
    selectedExpert,
    isLoadingRecipient,
    selectRecipient,
  } = useRecipientPicker();
  const isComposerDisabled = isCreatingSession || !!isInteractionLocked;
  const suggestions = useSuggestedTasks({
    surface: intro.isVisible ? "intro" : "home",
    onSend: onStartTask ?? onSend,
    enabled: !intro.isAwaitingGreeting,
    experts,
    expertsById,
    isRosterLoaded,
    hasRosterFailed,
    recipientID: recipient.id,
    selectRecipient,
    // A task starts its own chat only through onStartTask, which the page
    // withholds while it resolves the recipient or the turn is locked.
    canSend: Boolean(onStartTask) && !isComposerDisabled,
    isRecipientResolving: isLoadingRecipient,
  });
  const collapse = useCollapsedSuggestions();
  const introLine = isLoadingRecipient ? null : getIntroLine(selectedExpert);

  const recipientPicker = (
    <RecipientChip
      recipient={recipient}
      options={options}
      isLoading={isLoadingRecipient}
      onSelect={selectRecipient}
    />
  );

  const { inputPlaceholder } = useHomeComposer({
    enabled:
      !intro.isVisible && !intro.isAwaitingGreeting && !isKickoffStarting,
    greetingPath: intro.isVisible ? intro.path : null,
  });

  const [isInfoOpen, setIsInfoOpen] = useState(false);
  // The regular hero: its composer sits in the tray, which names the
  // recipient and the connection, and the expert's profile opens beside it.
  const isRegularHero = !intro.isVisible && !intro.isAwaitingGreeting;
  // Home's two tabs, held back with the hero so a load shows one loader.
  const [homeView, setHomeView] = useHomeView();

  if (isKickoffStarting) {
    return <ExpertKickoffLoader expertName={expertName} />;
  }

  if (isRegularHero && homeView === "tasks") {
    return (
      <div className="flex h-full min-h-0 w-full flex-1 flex-col gap-3 px-4 pb-4 md:px-6">
        <HomeViewTabs value={homeView} onChange={setHomeView} />
        <HomeBoard />
      </div>
    );
  }

  const composer = (
    <ChatInput
      inputId="chat-input-empty"
      draftKey={`new:${selectedExpert?.id ?? expertId ?? "default"}`}
      stacked
      voiceMode={voiceMode}
      modeSelector={modeSelector}
      onSend={onSend}
      disabled={isComposerDisabled}
      atReset={atReset}
      isUploadingFiles={isUploadingFiles}
      placeholder={
        atReset
          ? SEND_AT_RESET_PLACEHOLDER
          : selectedExpert
            ? getExpertInputPlaceholder(selectedExpert.name)
            : inputPlaceholder
      }
      className={cn(
        "w-full max-md:[&_[data-slot=input-group-control]]:min-h-16",
        "[&_textarea]:min-h-10 [&_textarea]:placeholder:text-foreground-faint",
      )}
      droppedFiles={droppedFiles}
      onDroppedFilesConsumed={onDroppedFilesConsumed}
      expertId={expertId}
      expertName={expertName}
      assistantName={selectedExpert?.name ?? expertName ?? null}
      recipientPicker={intro.isVisible ? recipientPicker : undefined}
    />
  );

  return (
    <div className="flex h-full min-w-0 flex-1">
      <div className="relative flex h-full min-w-0 flex-1 flex-col">
        {/* Behind everything on the Chat tab; the content sits above it. */}
        {isRegularHero && <DitherTrail />}
        {isRegularHero ? (
          <InfoToggle
            isPanelOpen={isInfoOpen}
            onOpen={() => setIsInfoOpen(true)}
          />
        ) : (
          // Which connection the new chat runs on, in the page corner while
          // the composer has no tray to carry it.
          <div className="absolute top-3 right-3 z-30 empty:hidden">
            <ConnectionPicker className="ml-0" />
          </div>
        )}
        {isRegularHero && (
          <HomeViewTabs value={homeView} onChange={setHomeView} />
        )}
        <HomeScroller>
          <motion.div
            className="relative z-10 my-auto w-full max-w-[1120px] text-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3 }}
          >
            <div className="mx-auto max-w-[52rem] pt-6">
              {intro.isVisible ? (
                <OnboardingIntroCard
                  name={greetingName}
                  greeting={intro.greeting}
                  suggestions={suggestions}
                  collapse={collapse}
                  transcript={intro.transcript}
                  disabled={isComposerDisabled || !!atReset}
                />
              ) : intro.isAwaitingGreeting ? (
                // For as long as the pipeline is still writing, which can be
                // behind the first-run walkthrough. The orb it renders is the
                // same element the card above puts in its heading, so the
                // swap moves it there rather than replacing it.
                intro.isFirstRunPending ? (
                  <GreetingLoader />
                ) : (
                  // An ordinary load: the page's own loader, in the same
                  // spot, so the two read as one.
                  <div
                    role="status"
                    className="pointer-events-none fixed inset-0 z-40 flex items-center justify-center"
                  >
                    <span className="sr-only">Loading</span>
                    <ScaleLoader className="text-foreground-subtle" />
                  </div>
                )
              ) : (
                // The mark, then the line that names who to tell.
                <>
                  <div className="mb-6 flex justify-center">
                    <AutoGPTLogo
                      hideText
                      viewBox="47 -1 42 42"
                      aria-label="AutoGPT"
                      isMono
                      className="size-12"
                    />
                  </div>
                  <EmptyHero
                    intro={introLine}
                    recipient={recipient}
                    isExpert={Boolean(selectedExpert)}
                    expertRole={selectedExpert?.role}
                  />
                </>
              )}

              {/* Held back while the greeting is on its way — it enters with
                  the greeting page instead of sitting under a bare hero. */}
              {!intro.isAwaitingGreeting && (
                <div className={cn("mb-6", intro.isVisible && "max-w-[48rem]")}>
                  <div
                    {...tourTarget("composer")}
                    className={cn(
                      "text-left transition-colors duration-300 ease-out",
                      // The greeting's prompt card bleeds 1.25rem past the
                      // text (-mx-5); the composer stretches the same amount
                      // so their edges line up. The regular hero keeps it
                      // centered.
                      intro.isVisible
                        ? "-mx-5 max-w-[50.5rem]"
                        : "mx-auto w-full max-w-[42rem]",
                    )}
                  >
                    {usageNotice && (
                      <div className="mb-3 text-left">{usageNotice}</div>
                    )}
                    {intro.isVisible ? (
                      composer
                    ) : (
                      <ComposerTray
                        recipientPicker={recipientPicker}
                        connection={
                          <ConnectionPicker withLabel className="ml-0" />
                        }
                      >
                        {composer}
                      </ComposerTray>
                    )}
                    {lowHint}
                  </div>
                </div>
              )}
            </div>
            {isRegularHero && <HomeRecentRow expertId={recipient.id} />}
          </motion.div>
        </HomeScroller>
      </div>
      {isRegularHero && (
        <ExpertInfoPanel
          recipient={recipient}
          isOpen={isInfoOpen}
          onClose={() => setIsInfoOpen(false)}
        />
      )}
    </div>
  );
}
