"use client";

import Link from "next/link";
import { Alert02Icon } from "@hugeicons/core-free-icons";

import { Icon } from "@/components/atoms/Icon/Icon";
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
    <div
      role="status"
      aria-label="No AI connection"
      className={cn(
        "mb-2 flex w-full items-start gap-2 rounded-2xl bg-amber-50 px-3 py-2 text-left text-sm text-amber-900",
        className,
      )}
    >
      <Icon
        icon={Alert02Icon}
        className="mt-0.5 h-4 w-4 shrink-0"
        aria-hidden
      />
      <span className="min-w-0 flex-1">
        No AI model is connected, so chat can&apos;t answer yet. Whoever runs
        this server can add a model API key or a local model{" "}
        <a
          href={SELF_HOSTED_MODELS_DOCS_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="font-medium underline underline-offset-2"
        >
          (setup guide)
        </a>
        , or you can connect your own ChatGPT or Microsoft 365 Copilot plan in{" "}
        <Link
          href="/settings/integrations"
          className="font-medium underline underline-offset-2"
        >
          Settings → Integrations
        </Link>
        .
      </span>
    </div>
  );
}
