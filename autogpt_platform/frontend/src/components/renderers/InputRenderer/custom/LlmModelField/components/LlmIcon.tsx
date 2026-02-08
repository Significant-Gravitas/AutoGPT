"use client";

import Image from "next/image";
import { Text } from "@/components/atoms/Text/Text";

const creatorIconMap: Record<string, string> = {
  anthropic: "/integrations/anthropic-color.png",
  openai: "/integrations/openai.png",
  google: "/integrations/gemini.png",
  nvidia: "/integrations/nvidia.png",
  groq: "/integrations/groq.png",
  ollama: "/integrations/ollama.png",
  openrouter: "/integrations/open_router.png",
  v0: "/integrations/v0.png",
  xai: "/integrations/xai.webp",
  meta: "/integrations/llama_api.png",
  amazon: "/integrations/amazon.png",
  cohere: "/integrations/cohere.png",
  deepseek: "/integrations/deepseek.png",
  gryphe: "/integrations/gryphe.png",
  microsoft: "/integrations/microsoft.webp",
  moonshotai: "/integrations/moonshot.png",
  mistral: "/integrations/mistral.png",
  mistralai: "/integrations/mistral.png",
  nousresearch: "/integrations/nousresearch.avif",
  perplexity: "/integrations/perplexity.webp",
  qwen: "/integrations/qwen.png",
};

type Props = {
  value: string;
  size?: number;
};

export function LlmIcon({ value, size = 20 }: Props) {
  const normalized = value.trim().toLowerCase().replace(/\s+/g, "");
  const src = creatorIconMap[normalized];
  if (src) {
    return (
      <div
        className="flex items-center justify-center overflow-hidden rounded-xsmall"
        style={{ width: size, height: size }}
      >
        <Image
          src={src}
          alt={value}
          width={size}
          height={size}
          className="h-full w-full object-cover"
        />
      </div>
    );
  }

  const fallback = value?.trim().slice(0, 1).toUpperCase() || "?";
  return (
    <div
      className="flex items-center justify-center rounded-xsmall bg-zinc-100"
      style={{ width: size, height: size }}
    >
      <Text variant="small" className="text-zinc-500">
        {fallback}
      </Text>
    </div>
  );
}
