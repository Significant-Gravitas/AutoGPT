"use client";

import Link from "next/link";
import { Alert02Icon } from "@hugeicons/core-free-icons";

import { Icon } from "@/components/atoms/Icon/Icon";
import { AttentionCard } from "@/components/organisms/AttentionCard/AttentionCard";
import { cn } from "@/lib/utils";

interface Props {
  visible: boolean;
  /** Who the message goes to: Otto, or the expert the chat is with. */
  assistantName: string;
  className?: string;
}

/** Shown while the draft holds something shaped like a key. It only warns:
 *  sending stays possible and the text is never changed. The status region
 *  stays mounted, hidden and empty, so screen readers announce the warning
 *  when it appears inside it; a region mounted with its text is often not
 *  announced. */
export function SecretPasteNotice({
  visible,
  assistantName,
  className,
}: Props) {
  return (
    <div
      role="status"
      data-testid="secret-paste-notice"
      className={visible ? cn("w-full", className) : "sr-only"}
    >
      {visible && (
        <AttentionCard.Root variant="warning" density="compact">
          <AttentionCard.Row>
            <AttentionCard.Icon>
              <Icon icon={Alert02Icon} className="h-4 w-4" aria-hidden />
            </AttentionCard.Icon>
            <AttentionCard.Content asChild>
              <span>
                This looks like an API key or other secret. Anything you send
                here is saved in the chat. To let {assistantName} use a key, add
                it under{" "}
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
            </AttentionCard.Content>
          </AttentionCard.Row>
        </AttentionCard.Root>
      )}
    </div>
  );
}
