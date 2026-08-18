"use client";

import { MessageAction } from "@/components/ai-elements/message";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useState } from "react";
import { Copy02Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  text: string;
}

export function CopyButton({ text }: Props) {
  const [copied, setCopied] = useState(false);

  if (!text.trim()) return null;

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      toast({
        title: "Failed to copy",
        description:
          "Your browser may not support clipboard access, or something went wrong.",
        variant: "destructive",
      });
    }
  }

  return (
    <MessageAction
      tooltip={copied ? "Copied!" : "Copy"}
      onClick={handleCopy}
      variant="ghost"
      size="icon-sm"
    >
      {copied ? (
        <Icon icon={Tick02Icon} size={16} />
      ) : (
        <Icon icon={Copy02Icon} size={16} />
      )}
    </MessageAction>
  );
}
