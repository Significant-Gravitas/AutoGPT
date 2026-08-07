"use client";

import { ChatInput } from "@/app/(platform)/copilot/components/ChatInput/ChatInput";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { motion } from "framer-motion";
import type { WorkspaceAttachment } from "../../helpers/workspaceAttachments";
import { EmptyHero } from "../EmptySession/components/EmptyHero";
import { getGreetingName } from "../EmptySession/helpers";
import { PulseChips } from "../PulseChips/PulseChips";
import { usePulseChips } from "../PulseChips/usePulseChips";
import { BriefingCard } from "./components/BriefingCard/BriefingCard";
import { useCopilotHome } from "./useCopilotHome";

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

export function CopilotHome({
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
  const pulseChips = usePulseChips();
  const { briefing, isLoadingBriefing, hasBriefing } = useCopilotHome();
  const isComposerDisabled = isCreatingSession || !!isAdoptingExpertSession;

  return (
    <div className="relative flex h-full flex-1 items-center justify-center overflow-y-auto px-0 py-5 md:px-6 md:py-10">
      <motion.div
        className="relative z-10 w-full max-w-[52rem] text-center"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3 }}
      >
        <div className="mx-auto max-w-[52rem]">
          <EmptyHero
            name={greetingName}
            isAwaitingGreeting={false}
            isGreetingFlow={false}
          />

          {/* Briefing card slot — falls back to the pulse strip while the
              user has no briefing yet; nothing renders until load settles
              so the strip doesn't flash for users who do have a briefing. */}
          {isLoadingBriefing ? null : hasBriefing && briefing ? (
            <BriefingCard briefing={briefing} />
          ) : (
            <PulseChips chips={pulseChips} onChipClick={onSend} />
          )}

          {/* Needs-attention slot (Task 6) */}

          {/* Team strip slot (Task 5) */}

          <div className="mb-6">
            <motion.div
              layoutId={inputLayoutId}
              transition={{ type: "spring", bounce: 0.2, duration: 0.65 }}
              className="w-full px-2"
            >
              <ChatInput
                inputId="chat-input-empty"
                onSend={onSend}
                disabled={isComposerDisabled}
                isUploadingFiles={isUploadingFiles}
                droppedFiles={droppedFiles}
                onDroppedFilesConsumed={onDroppedFilesConsumed}
                className="w-full"
              />
            </motion.div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
