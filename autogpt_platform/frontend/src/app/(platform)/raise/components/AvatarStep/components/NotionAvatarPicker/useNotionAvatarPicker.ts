import {
  clampPart,
  colorForToken,
  notionAvatarUrlFor,
  notionConfigForName,
  randomNotionConfig,
  type NotionAvatarConfig,
} from "@/components/molecules/NotionAvatar/helpers";
import type { NotionCategory } from "@/components/molecules/NotionAvatar/metadata.generated";
import { useState } from "react";

interface Args {
  name: string;
  color: string | null;
  onPick: (avatarUrl: string) => void;
}

export function useNotionAvatarPicker({ name, color, onPick }: Args) {
  const [config, setConfig] = useState<NotionAvatarConfig>(() =>
    seedConfig(name, color),
  );

  // The colour was answered a beat ago, so shuffling the face leaves it alone.
  // Marks stay off too: blush and freckles are something to opt into from the
  // row, not something a generated face should arrive wearing.
  function shuffle() {
    setConfig((current) => ({
      ...withoutMarks(randomNotionConfig()),
      color: current.color,
    }));
  }

  function cycle(category: NotionCategory, step: number) {
    setConfig((current) => ({
      ...current,
      parts: {
        ...current.parts,
        [category]: clampPart(category, current.parts[category] + step),
      },
    }));
  }

  function confirm() {
    onPick(notionAvatarUrlFor(config));
  }

  return { config, shuffle, cycle, confirm };
}

function seedConfig(name: string, color: string | null): NotionAvatarConfig {
  const seeded = withoutMarks(notionConfigForName(name));
  const token = colorForToken(color);
  return token ? { ...seeded, color: token } : seeded;
}

function withoutMarks(config: NotionAvatarConfig): NotionAvatarConfig {
  return { ...config, parts: { ...config.parts, details: 0 } };
}
