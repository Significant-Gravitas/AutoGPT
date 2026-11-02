import type { VoiceSample } from "@/app/api/__generated__/models/voiceSample";

export type VoicePickChoice = "a" | "b" | "custom";

export interface VoicePickResult {
  choice: VoicePickChoice;
  customText?: string;
}

// The result maps to the string stored in voice_preferences, which renders
// verbatim into the prompt's <voice_preferences> block — so it must read as
// plain guidance, never JSON. A preset pick keeps the chosen sample as a
// concrete example; a custom pick anchors on the user's own words.
//
// Returns null when the pick resolves to nothing storable (a choice pointing
// at a missing sample, or blank custom text) so callers can decline the save
// instead of silently overwriting the expert's voice with "".
export function buildVoicePreferences(
  result: VoicePickResult,
  samples: VoiceSample[],
): string | null {
  if (result.choice === "custom") {
    const text = (result.customText ?? "").trim();
    if (!text) return null;
    return `Preferred writing style: match the user's own writing sample below.\n\nExample to match:\n\n${text}`;
  }
  const sample = result.choice === "a" ? samples[0] : samples[1];
  if (!sample) return null;
  return `Preferred writing style: ${sample.label}.\n\nExample to match:\n\n${sample.text}`;
}

import { cn } from "@/lib/utils";

const SELECTABLE_CARD_CLASS_NAME =
  "border border-border bg-background transition-colors has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-ring";
const CARD_DENSITY = {
  regular: "rounded-2xl p-5",
  compact: "rounded-lg p-3",
};

export interface SelectableCardColors {
  /** Border, fill and ring once the option is chosen. */
  selected?: string;
  /** Hover border and focus ring; defaults lean on neutral tokens. */
  interactive?: string;
}

export function selectableCardClassName(
  isSelected: boolean,
  interactive = false,
  colors: SelectableCardColors = {},
  compact = false,
) {
  return cn(
    SELECTABLE_CARD_CLASS_NAME,
    compact ? CARD_DENSITY.compact : CARD_DENSITY.regular,
    colors.interactive ?? "focus-within:ring-ring",
    isSelected
      ? (colors.selected ?? "border-accent bg-accent/5 ring-2 ring-accent/20")
      : interactive &&
          cn(
            "cursor-pointer",
            colors.interactive ? null : "hover:border-foreground/30",
          ),
  );
}
