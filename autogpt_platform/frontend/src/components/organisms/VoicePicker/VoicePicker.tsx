"use client";

import type { VoiceSample } from "@/app/api/__generated__/models/voiceSample";
import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";
import { SampleCard } from "./components/SampleCard";
import type { VoicePickResult } from "./helpers";
import { useVoicePicker } from "./useVoicePicker";

interface Props {
  name?: string;
  samples: VoiceSample[];
  onPick: (result: VoicePickResult) => void;
  onSkip: () => void;
  isSubmitting?: boolean;
}

export function VoicePicker({
  name,
  samples,
  onPick,
  onSkip,
  isSubmitting = false,
}: Props) {
  const {
    selected,
    customText,
    selectSample,
    focusCustom,
    changeCustom,
    canSubmit,
    submit,
  } = useVoicePicker({ onPick });

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1.5">
        <h2 className="text-2xl font-semibold tracking-[-0.02em] text-zinc-900">
          {name ? `How should ${name} write?` : "How should this expert write?"}
        </h2>
        <p className="text-base text-zinc-500">
          Pick the voice that feels right. You can fine-tune it anytime in the
          Soul editor.
        </p>
      </header>

      <div className="flex flex-col gap-3">
        {samples.slice(0, 2).map((sample, index) => (
          <SampleCard
            key={sample.label}
            sample={sample}
            isSelected={selected === (index === 0 ? "a" : "b")}
            onSelect={() => selectSample(index === 0 ? "a" : "b")}
          />
        ))}

        <div
          className={cn(
            "rounded-2xl border p-5 transition-colors",
            selected === "custom"
              ? "border-purple-300 bg-purple-50/40 ring-2 ring-purple-200"
              : "border-zinc-200 bg-white",
          )}
        >
          <label
            htmlFor="voice-custom"
            className="mb-2 block text-xs font-medium uppercase tracking-[0.12em] text-purple-600"
          >
            Paste your own
          </label>
          <textarea
            id="voice-custom"
            value={customText}
            onFocus={focusCustom}
            onChange={(event) => changeCustom(event.target.value)}
            rows={3}
            maxLength={2000}
            placeholder="Paste a few sentences written the way you'd like this expert to sound."
            className="w-full resize-none rounded-xl border border-zinc-200 bg-white px-4 py-2.5 text-sm leading-relaxed text-black placeholder:text-zinc-500 focus:border-purple-400 focus:outline-none focus:ring-1 focus:ring-purple-400"
          />
        </div>
      </div>

      <footer className="flex items-center justify-between gap-3">
        <Button variant="ghost" onClick={onSkip} disabled={isSubmitting}>
          Skip for now
        </Button>
        <Button
          variant="primary"
          onClick={submit}
          disabled={!canSubmit}
          loading={isSubmitting}
          className="rounded-full"
        >
          Use this voice
        </Button>
      </footer>
    </div>
  );
}
