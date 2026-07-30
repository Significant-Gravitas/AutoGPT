"use client";

import { ChatInput } from "@/app/(platform)/copilot/components/ChatInput/ChatInput";
import { useGetV2GetSuggestedPrompts } from "@/app/api/__generated__/endpoints/chat/chat";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { TextGenerateEffect } from "@/components/ui/text-generate-effect";
import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import { getSuggestionThemes, INPUT_PLACEHOLDER } from "./helpers";
import { PERSONAS } from "./personas";
import { PersonaAvatar } from "./components/PersonaAvatar";
import { PersonaDial } from "./components/PersonaDial/PersonaDial";
import { SuggestionThemes } from "./components/SuggestionThemes/SuggestionThemes";
import type { WorkspaceAttachment } from "../../helpers/workspaceAttachments";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
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
  const [personaIndex, setPersonaIndex] = useState(0);
  const [isDialOpen, setIsDialOpen] = useState(false);
  const persona = PERSONAS[personaIndex];
  const isExpertsEnabled = useGetFlag(Flag.HIRE_EXPERTS);
  const { options, recipient, isLoadingRecipient, selectRecipient } =
    useRecipientPicker();
  const isComposerDisabled = isCreatingSession || !!isAdoptingExpertSession;

  // Persona is mirrored in the URL (?persona=id) so it survives reloads and
  // can be shared. Read after mount to avoid SSR/client hydration mismatch.
  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get("persona");
    const index = PERSONAS.findIndex((p) => p.id === id);
    if (index >= 0) setPersonaIndex(index);
  }, []);

  function handleSelectPersona(index: number) {
    setPersonaIndex(index);
    const url = new URL(window.location.href);
    url.searchParams.set("persona", PERSONAS[index].id);
    window.history.replaceState(null, "", url);
  }

  const { data: suggestedPromptsResponse, isLoading: isLoadingPrompts } =
    useGetV2GetSuggestedPrompts({
      query: { staleTime: Infinity, gcTime: Infinity, refetchOnMount: false },
    });
  const themes = getSuggestionThemes(
    suggestedPromptsResponse?.status === 200
      ? suggestedPromptsResponse.data.themes
      : undefined,
  );

  return (
    <div className="relative flex h-full flex-1 items-center justify-center overflow-y-auto overflow-x-hidden px-0 pb-24 pt-5 md:px-6 md:pb-32 md:pt-10">
      <motion.div
        className="relative z-10 w-full max-w-[52rem] text-center"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3 }}
      >
        <div className="mx-auto max-w-[52rem]">
          <div className="relative" data-persona-picker>
            {isDialOpen && (
              <PersonaDial
                personas={PERSONAS}
                selectedIndex={personaIndex}
                onSelect={handleSelectPersona}
                onClose={() => setIsDialOpen(false)}
              />
            )}
            <PersonaAvatar
              persona={persona}
              isOpen={isDialOpen}
              onToggle={() => setIsDialOpen((open) => !open)}
            />
          </div>
          <Text variant="h3" className="mb-1 !text-[1.375rem] text-zinc-700">
            Hi, I am{" "}
            <span style={{ color: persona.accent }}>{persona.name}</span>, your{" "}
            {persona.role.toLowerCase()}
          </Text>
          <TextGenerateEffect
            className="mb-8 !font-normal [&>div]:!mt-0 [&_div]:!text-[1.375rem] [&_div]:!leading-normal [&_div]:!tracking-normal"
            duration={0.6}
            words="What can I do for you today?"
          />

          <div className="mb-6 mt-10">
            <motion.div
              layoutId={inputLayoutId}
              transition={{ type: "spring", bounce: 0.2, duration: 0.65 }}
              className="mx-auto w-full max-w-[42rem] overflow-hidden rounded-xlarge border transition-colors duration-300 ease-out"
              style={{
                borderColor: `${persona.accent}55`,
                boxShadow: `0 2px 8px rgba(0,0,0,0.04), 0 0 32px -4px ${persona.accent}59`,
              }}
            >
              <ChatInput
                inputId="chat-input-empty"
                onSend={onSend}
                disabled={isComposerDisabled}
                hideSubmitWhenEmpty
                isUploadingFiles={isUploadingFiles}
                placeholder={INPUT_PLACEHOLDER}
                className="w-full [&_textarea]:min-h-[4.5rem]"
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
            disabled={isComposerDisabled}
          />
        )}
      </motion.div>
    </div>
  );
}
