"use client";

import { Button } from "@/components/atoms/Button/Button";
import { VoicePicker } from "@/components/organisms/VoicePicker/VoicePicker";
import { VOICE_SAMPLES } from "../../helpers";
import { useRaisePage } from "../../useRaisePage";
import { AssistantBubble } from "../AssistantBubble/AssistantBubble";
import { FirstJobStep } from "../FirstJobStep/FirstJobStep";
import { NameStep } from "../NameStep/NameStep";
import { SoulPreviewPanel } from "../SoulPreviewPanel/SoulPreviewPanel";

export function RaiseFlow() {
  const {
    step,
    messages,
    name,
    voiceLabel,
    firstJob,
    isSubmitting,
    submitName,
    pickVoice,
    skipVoice,
    pickFirstJob,
    skipFirstJob,
    finish,
  } = useRaisePage();

  return (
    <main className="min-h-screen bg-zinc-50 px-4 pb-16 pt-6 sm:px-6 lg:px-8">
      <div className="mx-auto grid w-full max-w-[1000px] gap-6 lg:grid-cols-[1fr_minmax(300px,360px)]">
        <div className="order-2 flex flex-col gap-4 lg:order-1">
          <div className="flex flex-col gap-3">
            {messages.map((message) =>
              message.role === "assistant" ? (
                <AssistantBubble key={message.id} text={message.text} />
              ) : (
                <div
                  key={message.id}
                  className="max-w-[80%] self-end rounded-3xl rounded-br-lg bg-purple-600 px-5 py-3.5 text-[15px] leading-relaxed text-white"
                >
                  {message.text}
                </div>
              ),
            )}
          </div>

          <div className="mt-2">
            {step === "name" ? <NameStep onSubmit={submitName} /> : null}
            {step === "voice" ? (
              <VoicePicker
                name={name}
                samples={VOICE_SAMPLES}
                onPick={pickVoice}
                onSkip={skipVoice}
              />
            ) : null}
            {step === "firstJob" ? (
              <FirstJobStep onPick={pickFirstJob} onSkip={skipFirstJob} />
            ) : null}
            {step === "review" ? (
              <div className="flex justify-end">
                <Button
                  variant="primary"
                  onClick={finish}
                  loading={isSubmitting}
                  className="rounded-full"
                >
                  {`Bring ${name} to life`}
                </Button>
              </div>
            ) : null}
          </div>
        </div>

        <div className="order-1 lg:order-2">
          <SoulPreviewPanel
            name={name}
            voiceLabel={voiceLabel}
            firstJobName={firstJob?.name ?? null}
          />
        </div>
      </div>
    </main>
  );
}
