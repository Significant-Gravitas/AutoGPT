import { useCreateRaisedExpert } from "@/app/api/__generated__/endpoints/experts/experts";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { VoicePickResult } from "@/components/organisms/VoicePicker/helpers";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useRouter } from "next/navigation";
import { useState } from "react";
import {
  RAISE_PROMPTS,
  RAISE_STEPS,
  resolveVoicePreferences,
  voiceSummaryLabel,
  VOICE_SAMPLES,
  type RaiseStep,
} from "./helpers";

interface RaiseMessage {
  id: string;
  role: "assistant" | "user";
  text: string;
}

interface FirstJob {
  id: string;
  name: string;
}

export function useRaisePage() {
  const router = useRouter();
  const { mutateAsync: createRaisedExpert, isPending } =
    useCreateRaisedExpert();

  const [step, setStep] = useState<RaiseStep>("name");
  const [messages, setMessages] = useState<RaiseMessage[]>([
    { id: "assistant-name", role: "assistant", text: RAISE_PROMPTS.name },
  ]);
  const [name, setName] = useState("");
  const [voicePreferences, setVoicePreferences] = useState("");
  const [voiceLabel, setVoiceLabel] = useState<string | null>(null);
  const [firstJob, setFirstJob] = useState<FirstJob | null>(null);

  function advance(userText: string, next: RaiseStep, prompt: string) {
    setMessages((prev) => [
      ...prev,
      { id: `user-${step}`, role: "user", text: userText },
      { id: `assistant-${next}`, role: "assistant", text: prompt },
    ]);
    setStep(next);
  }

  function submitName(value: string) {
    const trimmed = value.trim();
    if (!trimmed) return;
    setName(trimmed);
    advance(trimmed, "voice", RAISE_PROMPTS.voice(trimmed));
  }

  function pickVoice(result: VoicePickResult) {
    setVoicePreferences(resolveVoicePreferences(result));
    const label = voiceSummaryLabel(result, VOICE_SAMPLES);
    setVoiceLabel(label);
    advance(label, "firstJob", RAISE_PROMPTS.firstJob);
  }

  function skipVoice() {
    advance("I'll decide the voice later", "firstJob", RAISE_PROMPTS.firstJob);
  }

  function pickFirstJob(job: FirstJob) {
    setFirstJob(job);
    advance(job.name, "review", RAISE_PROMPTS.review);
  }

  function skipFirstJob() {
    advance("Skip for now", "review", RAISE_PROMPTS.review);
  }

  async function finish() {
    try {
      const response = await createRaisedExpert({
        data: {
          name,
          voice_preferences: voicePreferences || null,
          first_job_store_listing_version_id: firstJob?.id ?? null,
        },
      });
      const expert = response.data as Expert;
      router.push(`/copilot?expertId=${expert.id}&kickoff=1`);
    } catch {
      toast({
        title: `Couldn't raise ${name || "your expert"}`,
        description: "Something went wrong. Please try again.",
        variant: "destructive",
      });
    }
  }

  return {
    steps: RAISE_STEPS,
    step,
    messages,
    name,
    voiceLabel,
    firstJob,
    isSubmitting: isPending,
    submitName,
    pickVoice,
    skipVoice,
    pickFirstJob,
    skipFirstJob,
    finish,
  };
}
