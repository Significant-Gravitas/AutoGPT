import {
  getGetExpertQueryKey,
  useUpdateExpertSoul,
} from "@/app/api/__generated__/endpoints/experts/experts";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { toast } from "@/components/molecules/Toast/use-toast";
import {
  buildVoicePreferences,
  type VoicePickResult,
} from "@/components/organisms/VoicePicker/helpers";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { VOICE_SAMPLES } from "./helpers";

export function useExpertSetupSection(expert: Expert) {
  const queryClient = useQueryClient();
  const [isBudgetOpen, setIsBudgetOpen] = useState(false);
  const [isVoiceOpen, setIsVoiceOpen] = useState(false);
  const { mutateAsync: updateSoul, isPending: isSavingVoice } =
    useUpdateExpertSoul();

  async function pickVoice(result: VoicePickResult) {
    const voicePreferences = buildVoicePreferences(result, VOICE_SAMPLES);
    if (voicePreferences === null) {
      setIsVoiceOpen(false);
      return;
    }
    try {
      await updateSoul({
        expertId: expert.id,
        data: {
          name: expert.name,
          identity: expert.identity,
          voice_preferences: voicePreferences,
          boundaries: expert.boundaries,
        },
      });
      await queryClient.invalidateQueries({
        queryKey: getGetExpertQueryKey(expert.id),
      });
      setIsVoiceOpen(false);
    } catch {
      toast({ title: "Couldn't save the voice", variant: "destructive" });
    }
  }

  return {
    isBudgetOpen,
    openBudget: () => setIsBudgetOpen(true),
    closeBudget: () => setIsBudgetOpen(false),
    isVoiceOpen,
    setIsVoiceOpen,
    isSavingVoice,
    pickVoice,
  };
}
