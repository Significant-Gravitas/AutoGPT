"use client";

import { ChatInput } from "@/app/(platform)/copilot/components/ChatInput/ChatInput";
// TODO: restore HomeRecap
// import { HomeRecap } from "@/app/(platform)/home/components/HomeRecap/HomeRecap";
import { tourTarget } from "@/lib/tour-targets";
import { useFirstRunTourAutoStart } from "@/app/(platform)/components/FirstRunTour/useFirstRunTourAutoStart";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { GlimmProvider } from "glimm/react";
import { useState, type ComponentProps, type ReactNode } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { SidebarRightIcon } from "@hugeicons/core-free-icons";
import type { WorkspaceAttachment } from "../../helpers/workspaceAttachments";
import { ConnectionPicker } from "../ChatInput/components/ConnectionPicker/ConnectionPicker";
import { RecipientChip } from "../ChatInput/components/RecipientChip";
import { OnboardingIntroCard } from "../OnboardingIntroCard/OnboardingIntroCard";
import { useOnboardingIntroCard } from "../OnboardingIntroCard/useOnboardingIntroCard";
import { SEND_AT_RESET_PLACEHOLDER } from "../SendAtReset/helpers";
// import { SuggestedTasksBlock } from "../SuggestedTasks/SuggestedTasksBlock";
import { useSuggestedTasks } from "../SuggestedTasks/useSuggestedTasks";
import { isTasksV1Enabled } from "@/app/(platform)/tasks/data/flag";
import { TasksWorkspace } from "@/app/(platform)/tasks/components/TasksWorkspace/TasksWorkspace";
import { EmptyHero } from "./components/EmptyHero";
import { ExpertKickoffLoader } from "./components/ExpertKickoffLoader/ExpertKickoffLoader";
import { GreetingLoader } from "./components/GreetingLoader";
import {
  HomeViewSwitch,
  HomeViewTabs,
} from "./components/HomeViewTabs/HomeViewTabs";
import { useHomeView } from "./components/HomeViewTabs/useHomeView";
import { ComposerTray } from "./components/ComposerTray/ComposerTray";
import { HomeRecentChats } from "./components/HomeRecentChats/HomeRecentChats";
import { ExpertInfoPanel } from "./components/ExpertInfoPanel/ExpertInfoPanel";
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

export function EmptySession(props: Props) {
  return (
    <GlimmProvider sweepMs={900} midpoint={0.5}>
      <EmptySessionView {...props} />
    </GlimmProvider>
  );
}

function EmptySessionView({
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
  });
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

  const [homeView, setHomeView] = useHomeView();
  const showTabs = isTasksV1Enabled() && !intro.isVisible;
  const showInfoToggle = !intro.isVisible && !intro.isAwaitingGreeting;
  const [isInfoOpen, setIsInfoOpen] = useState(false);

  if (isKickoffStarting) {
    return <ExpertKickoffLoader expertName={expertName} />;
  }

  if (showTabs && homeView === "board") {
    return (
      <TasksWorkspace
        className="flex h-full min-h-0 w-full flex-1 flex-col px-4 pb-4 md:px-6"
        defaultView="board"
        openCreateFromURL={false}
        viewSwitch={<HomeViewSwitch value={homeView} onChange={setHomeView} />}
      />
    );
  }

  return (
    <div className="flex h-full min-w-0 flex-1">
      <div className="relative flex h-full min-w-0 flex-1 flex-col">
        {showInfoToggle && (
          <div
            inert={isInfoOpen}
            className={cn(
              "absolute top-4 right-4 z-40 transition-opacity",
              isInfoOpen
                ? "opacity-0 duration-200"
                : "opacity-100 delay-150 duration-300",
            )}
          >
            <Button
              type="button"
              variant="ghost-muted"
              size="icon-sm"
              leadingIcon={SidebarRightIcon}
              aria-label="Show expert info"
              className="size-7"
              onClick={() => setIsInfoOpen(true)}
            />
          </div>
        )}
        {showTabs && <HomeViewTabs value={homeView} onChange={setHomeView} />}
        <div className="relative flex min-h-0 flex-1 items-start justify-center overflow-y-auto px-0 py-5 md:px-6 md:py-10">
          <motion.div
            className={cn(
              "relative z-10 my-auto w-full text-center",
              "max-w-[1120px]",
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
                  suggestions={suggestions}
                  transcript={intro.transcript}
                  disabled={isComposerDisabled || !!atReset}
                />
              ) : intro.isAwaitingGreeting ? (
                // For as long as the pipeline is still writing, which can be
                // behind the first-run walkthrough. The orb it renders is the same
                // element the card above puts in its heading, so the swap
                // moves it there rather than replacing it.
                <GreetingLoader />
              ) : (
                <EmptyHero
                  name={greetingName}
                  intro={introLine}
                  recipient={recipient}
                  isExpert={Boolean(selectedExpert)}
                  expertRole={selectedExpert?.role}
                />
              )}

              {/* Held back while the greeting is on its way — it enters with
              the greeting page instead of sitting under a bare hero. */}
              {!intro.isAwaitingGreeting && (
                <div className={cn("mb-6", intro.isVisible && "max-w-[48rem]")}>
                  <div
                    {...tourTarget("composer")}
                    className={cn(
                      "text-left transition-colors duration-300 ease-out",
                      // The greeting's prompt card bleeds 1.25rem past the text
                      // (-mx-5); the composer stretches the same amount so their
                      // edges line up. No chrome of its own: the composer card is
                      // the only outline on screen. The regular hero keeps it
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

              {!intro.isVisible && !intro.isAwaitingGreeting && (
                <HomeRecentChats expertId={recipient.id} />
              )}

              {/* TODO: restore the suggested tasks under the composer.
            {!intro.isVisible &&
              !intro.isAwaitingGreeting &&
              !selectedExpert && (
                <SuggestedTasksBlock
                  suggestions={suggestions}
                  disabled={isComposerDisabled || !!atReset}
                />
              )} */}
            </div>

            {/* {!intro.isVisible && !intro.isAwaitingGreeting && <HomeRecap />} */}
          </motion.div>
        </div>
      </div>
      {showInfoToggle && (
        <ExpertInfoPanel
          recipient={recipient}
          isOpen={isInfoOpen}
          onClose={() => setIsInfoOpen(false)}
        />
      )}
    </div>
  );
}
