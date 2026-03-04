"use client";

import { MessageAction } from "@/components/ai-elements/message";
import { Check, Copy } from "@phosphor-icons/react";
import { useState } from "react";

interface Props {
  text: string;
}

export function CopyButton({ text }: Props) {
  const [copied, setCopied] = useState(false);

  if (!text.trim()) return null;

  async function handleCopy() {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  return (
    <MessageAction
      tooltip={copied ? "Copied!" : "Copy to clipboard"}
      onClick={handleCopy}
    >
      {copied ? <Check size={16} /> : <Copy size={16} />}
    </MessageAction>
  );
}
