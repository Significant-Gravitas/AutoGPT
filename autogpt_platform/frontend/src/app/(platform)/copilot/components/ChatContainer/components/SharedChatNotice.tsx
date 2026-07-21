"use client";

import { useState } from "react";
import { CheckIcon, CopyIcon } from "@phosphor-icons/react";
import { useGetV2GetChatShareState } from "@/app/api/__generated__/endpoints/chat/chat";
import { Button } from "@/components/atoms/Button/Button";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { chatShareUrl } from "@/lib/share/routes";

interface Props {
  sessionId: string;
}

export function SharedChatNotice({ sessionId }: Props) {
  const { toast } = useToast();
  const [copied, setCopied] = useState(false);
  const { data: shareState } = useGetV2GetChatShareState(sessionId, {
    query: {
      enabled: !!sessionId,
      select: (res) => (res.status === 200 ? res.data : undefined),
    },
  });

  if (!shareState?.is_shared || !shareState.share_token) return null;

  const shareUrl = chatShareUrl(shareState.share_token);

  async function copyShareUrl() {
    try {
      await navigator.clipboard.writeText(shareUrl);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      toast({
        title: "Failed to copy link",
        variant: "destructive",
      });
    }
  }

  return (
    <div className="mb-2 flex items-center justify-between gap-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
      <span className="font-medium">This chat is shared</span>
      <Button
        size="small"
        variant="secondary"
        onClick={copyShareUrl}
        leftIcon={
          copied ? (
            <CheckIcon size={14} weight="bold" />
          ) : (
            <CopyIcon size={14} />
          )
        }
      >
        {copied ? "Copied" : "Copy link"}
      </Button>
    </div>
  );
}
