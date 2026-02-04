"use client";

import {
  getGreetingName,
  getQuickActions,
} from "@/app/(platform)/copilot/helpers";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { ChatInput } from "@/components/contextual/Chat/components/ChatInput/ChatInput";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { SparkleIcon, SpinnerGapIcon } from "@phosphor-icons/react";
import { motion } from "framer-motion";
import { useState } from "react";

interface Props {
  inputLayoutId: string;
  isCreatingSession: boolean;
  onCreateSession: () => void | Promise<string>;
  onSend: (message: string) => void | Promise<void>;
}

export function EmptySession({
  inputLayoutId,
  isCreatingSession,
  onSend,
}: Props) {
  const { user } = useSupabase();
  const greetingName = getGreetingName(user);
  const quickActions = getQuickActions();
  const [loadingAction, setLoadingAction] = useState<string | null>(null);

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
    <div className="relative flex h-full flex-1 flex-col items-center justify-center overflow-hidden px-6 py-10 dark:bg-background">
      <motion.div
        className="relative w-full max-w-3xl"
        initial={{ opacity: 0, y: 14, filter: "blur(6px)" }}
        animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
        transition={{ type: "spring", bounce: 0.2, duration: 0.7 }}
      >
        <div className="mx-auto flex flex-col items-center text-center">
          <div className="mb-5 flex items-center gap-2 rounded-full border border-border/60 bg-white/70 px-3 py-1 text-xs text-muted-foreground shadow-sm backdrop-blur dark:bg-neutral-950/40">
            <SparkleIcon className="h-3.5 w-3.5 text-purple-600" />
            <span>Autopilot runs for you 24/7</span>
          </div>

          <Text variant="h3" className="mb-3 !text-[1.375rem] text-zinc-700">
            Hey, <span className="text-violet-600">{greetingName}</span>
          </Text>
          <Text variant="h3" className="!font-normal">
            What do you want to automate?
          </Text>
        </div>

        <div className="mx-auto p-5 dark:bg-neutral-950/40">
          <motion.div
            layoutId={inputLayoutId}
            transition={{ type: "spring", bounce: 0.2, duration: 0.65 }}
            className="w-full"
          >
            <ChatInput
              inputId="chat-input-empty"
              onSend={onSend}
              disabled={isCreatingSession}
              placeholder='You can search or just ask - e.g. "create a blog post outline"'
              className="w-full"
            />
          </motion.div>

          <div className="mt-8 flex w-full flex-nowrap items-center justify-center gap-3 [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
            {quickActions.map((action) => (
              <Button
                key={action}
                type="button"
                variant="outline"
                size="small"
                onClick={() => void handleQuickActionClick(action)}
                disabled={isCreatingSession || loadingAction !== null}
                aria-busy={loadingAction === action}
                leftIcon={
                  loadingAction === action ? (
                    <SpinnerGapIcon
                      className="h-4 w-4 animate-spin"
                      weight="bold"
                    />
                  ) : null
                }
                className="h-auto shrink-0 border-zinc-600 !px-4 !py-2 text-[1rem] text-zinc-600"
              >
                {action}
              </Button>
            ))}
          </div>
        </div>
      </motion.div>
    </div>
  );
}
