"use client";

import { ChatInput } from "@/app/(platform)/copilot/components/ChatInput/ChatInput";
import { useGetV2GetSuggestedPrompts } from "@/app/api/__generated__/endpoints/chat/chat";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import {
  getGreetingName,
  getInputPlaceholder,
  getSuggestionThemes,
} from "./helpers";
import { SuggestionThemes } from "./components/SuggestionThemes/SuggestionThemes";
import { PulseChips } from "../PulseChips/PulseChips";
import { usePulseChips } from "../PulseChips/usePulseChips";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { EditNameDialog } from "./components/EditNameDialog/EditNameDialog";

interface Props {
  inputLayoutId: string;
  isCreatingSession: boolean;
  onCreateSession: () => void | Promise<string>;
  onSend: (message: string, files?: File[]) => void | Promise<void>;
  isUploadingFiles?: boolean;
  droppedFiles?: File[];
  onDroppedFilesConsumed?: () => void;
}

export function EmptySession({
  inputLayoutId,
  isCreatingSession,
  onSend,
  isUploadingFiles,
  droppedFiles,
  onDroppedFilesConsumed,
}: Props) {
  const { user } = useSupabase();
  const greetingName = getGreetingName(user);
  const isAgentBriefingEnabled = useGetFlag(Flag.AGENT_BRIEFING);
  const pulseChips = usePulseChips();

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
    <div className="flex h-full flex-1 items-center justify-center overflow-y-auto bg-[#f8f8f9] px-0 py-5 md:px-6 md:py-10">
      <motion.div
        className="w-full max-w-[52rem] text-center"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3 }}
      >
        <div className="mx-auto max-w-[52rem]">
          <Text variant="h3" className="mb-1 !text-[1.375rem] text-zinc-700">
            Hey, <span className="text-violet-600">{greetingName}</span>
            <EditNameDialog currentName={greetingName} />
          </Text>
          <Text variant="h3" className="mb-8 !font-normal">
            Tell me about your work — I&apos;ll find what to automate.
          </Text>

          {isAgentBriefingEnabled && (
            <PulseChips chips={pulseChips} onChipClick={onSend} />
          )}

          <div className="mb-6">
            <motion.div
              layoutId={inputLayoutId}
              transition={{ type: "spring", bounce: 0.2, duration: 0.65 }}
              className="w-full px-2"
            >
              <ChatInput
                inputId="chat-input-empty"
                onSend={onSend}
                disabled={isCreatingSession}
                isUploadingFiles={isUploadingFiles}
                placeholder={inputPlaceholder}
                className="w-full"
                droppedFiles={droppedFiles}
                onDroppedFilesConsumed={onDroppedFilesConsumed}
              />
            </motion.div>
          </div>
        </div>

        {isLoadingPrompts ? (
          <div className="flex flex-wrap items-center justify-center gap-3">
            {Array.from({ length: 4 }, (_, i) => (
              <Skeleton key={i} className="h-10 w-28 shrink-0 rounded-full" />
            ))}
          </div>
        ) : (
          <SuggestionThemes
            themes={themes}
            onSend={onSend}
            disabled={isCreatingSession}
          />
        )}
      </motion.div>
    </div>
  );
}
