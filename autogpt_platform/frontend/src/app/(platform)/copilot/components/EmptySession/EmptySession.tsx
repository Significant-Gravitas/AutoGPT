"use client";

import { ChatInput } from "@/app/(platform)/copilot/components/ChatInput/ChatInput";

import { Text } from "@/components/atoms/Text/Text";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { Sparkle, SpinnerGapIcon } from "@phosphor-icons/react";
import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import {
  getGreetingName,
  getInputPlaceholder,
  getQuickActions,
} from "./helpers";

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
  const quickActions = getQuickActions();
  const [loadingAction, setLoadingAction] = useState<string | null>(null);
  const [inputPlaceholder, setInputPlaceholder] = useState(
    getInputPlaceholder(),
  );

  useEffect(() => {
    setInputPlaceholder(getInputPlaceholder(window.innerWidth));
  }, [window.innerWidth]);

  async function handleQuickActionClick(action: string) {
    if (isCreatingSession || loadingAction) return;

    setLoadingAction(action);
    try {
      await onSend(action);
    } finally {
      setLoadingAction(null);
    }
  }

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
          </Text>
          <Text variant="h3" className="mb-8 !font-normal">
            Tell me about your work — I&apos;ll find what to automate.
          </Text>

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

        <div className="mx-auto flex flex-col items-center divide-y divide-zinc-100 px-2">
          {quickActions.map((action) => (
            <button
              key={action}
              type="button"
              onClick={() => void handleQuickActionClick(action)}
              disabled={isCreatingSession || loadingAction !== null}
              aria-busy={loadingAction === action}
              className="flex items-center gap-2.5 px-1 py-3 text-[.9rem] text-zinc-600 transition-colors hover:text-zinc-900 disabled:opacity-50"
            >
              {loadingAction === action ? (
                <SpinnerGapIcon
                  className="h-4 w-4 shrink-0 animate-spin text-amber-400"
                  weight="bold"
                />
              ) : (
                <Sparkle
                  className="h-4 w-4 shrink-0 text-amber-400"
                  weight="fill"
                />
              )}
              {action}
            </button>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
