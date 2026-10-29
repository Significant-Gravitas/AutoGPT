"use client";

import { useState } from "react";
import { useGetV2GetChatShareState } from "@/app/api/__generated__/endpoints/chat/chat";
import { Button } from "@/components/atoms/Button/Button";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { chatShareUrl } from "@/lib/share/routes";
import { Copy01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { AttentionCard } from "@/components/organisms/AttentionCard/AttentionCard";

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
    <AttentionCard.Root
      variant="warning-outline"
      density="cozy"
      className="mb-2"
    >
      <AttentionCard.Row align="center" justify="between">
        <span className="font-medium">This chat is shared</span>
        <Button
          size="small"
          variant="secondary"
          onClick={copyShareUrl}
          leftIcon={
            copied ? (
              <Icon icon={Tick02Icon} size={14} />
            ) : (
              <Icon icon={Copy01Icon} size={14} />
            )
          }
        >
          {copied ? "Copied" : "Copy link"}
        </Button>
      </AttentionCard.Row>
    </AttentionCard.Root>
  );
}
