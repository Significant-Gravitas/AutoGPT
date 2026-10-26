"use client";

import Link from "next/link";
import { Alert02Icon } from "@hugeicons/core-free-icons";

import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";

interface Props {
  className?: string;
}

/** Shown while the draft holds something shaped like a key. It only warns:
 *  sending stays possible and the text is never changed. */
export function SecretPasteNotice({ className }: Props) {
  return (
    <div
      role="status"
      className={cn(
        "flex w-full items-start gap-1.5 rounded-2xl bg-amber-50 px-3 py-1.5 text-left text-sm text-amber-800",
        className,
      )}
    >
      <Icon
        icon={Alert02Icon}
        className="mt-0.5 h-4 w-4 shrink-0"
        aria-hidden
      />
      <span className="min-w-0 flex-1">
        This looks like an API key or other secret. Anything you send here is
        saved in the chat. To let Otto use a key, add it under{" "}
        <Link
          href="/settings/integrations"
          target="_blank"
          rel="noopener noreferrer"
          className="font-medium underline underline-offset-2"
        >
          Integrations
        </Link>{" "}
        instead.
      </span>
    </div>
  );
}
