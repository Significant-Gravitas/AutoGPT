"use client";

import Image from "next/image";
import { Text } from "@/components/atoms/Text/Text";
import { toLlmDisplayName } from "../helpers";
import claudeImg from "@/components/atoms/LLMItem/assets/claude.svg";
import gptImg from "@/components/atoms/LLMItem/assets/gpt.svg";
import perplexityImg from "@/components/atoms/LLMItem/assets/perplexity.svg";

const iconMap: Record<string, string> = {
  anthropic: claudeImg.src,
  claude: claudeImg.src,
  openai: gptImg.src,
  gpt: gptImg.src,
  open_router: gptImg.src,
  groq: gptImg.src,
  ollama: gptImg.src,
  llama_api: gptImg.src,
  perplexity: perplexityImg.src,
};

type Props = {
  value: string;
  size?: number;
};

export function LlmIcon({ value, size = 20 }: Props) {
  const normalized = value.toLowerCase();
  const src = iconMap[normalized];
  if (src) {
    return (
      <Image
        src={src}
        alt={toLlmDisplayName(value)}
        width={size}
        height={size}
        className="rounded-xsmall"
      />
    );
  }

  const fallback = toLlmDisplayName(value).slice(0, 1) || "?";
  return (
    <div className="flex h-5 w-5 items-center justify-center rounded-xsmall bg-zinc-100">
      <Text variant="small" className="text-zinc-500">
        {fallback}
      </Text>
    </div>
  );
}
