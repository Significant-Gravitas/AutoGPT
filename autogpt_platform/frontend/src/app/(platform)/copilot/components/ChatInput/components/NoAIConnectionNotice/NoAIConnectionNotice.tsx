"use client";

import Link from "next/link";
import { Alert02Icon } from "@hugeicons/core-free-icons";

import { Icon } from "@/components/atoms/Icon/Icon";
import { AttentionCard } from "@/components/organisms/AttentionCard/AttentionCard";
import { cn } from "@/lib/utils";
import { SELF_HOSTED_MODELS_DOCS_URL } from "@/app/(platform)/copilot/helpers/reportedSendError";
import { useNoAIConnectionNotice } from "./useNoAIConnectionNotice";

interface Props {
  enabled: boolean;
  className?: string;
}

// Stays above the composer for as long as nothing can answer, so the fix is
// still on screen after any toast about a failed send has gone.
export function NoAIConnectionNotice({ enabled, className }: Props) {
  const { hasNoConnection } = useNoAIConnectionNotice(enabled);
  if (!hasNoConnection) return null;

  return (
    <AttentionCard.Root
      variant="warning-strong"
      density="snug"
      role="status"
      aria-label="No AI connection"
      className={cn("mb-2 w-full", className)}
    >
      <AttentionCard.Row align="start">
        <AttentionCard.Icon nudge>
          <Icon icon={Alert02Icon} className="h-4 w-4" aria-hidden />
        </AttentionCard.Icon>
        <AttentionCard.Content asChild>
          <span>
            No AI model is connected, so chat can&apos;t answer yet. Whoever
            runs this server can add a model API key or a local model{" "}
            <a
              href={SELF_HOSTED_MODELS_DOCS_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="font-medium underline underline-offset-2"
            >
              (setup guide)
            </a>
            , or you can connect your own ChatGPT or Microsoft 365 Copilot plan
            in{" "}
            <Link
              href="/settings/integrations"
              className="font-medium underline underline-offset-2"
            >
              Settings → Integrations
            </Link>
            .
          </span>
        </AttentionCard.Content>
      </AttentionCard.Row>
    </AttentionCard.Root>
  );
}
