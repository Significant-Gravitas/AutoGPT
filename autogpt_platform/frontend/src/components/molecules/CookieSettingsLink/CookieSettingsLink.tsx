"use client";

import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import {
  isConsentManagerConfigured,
  openConsentSettings,
} from "@/services/consent/consent";
import { useConsentManagerStatus } from "@/services/consent/useConsent";

interface Props {
  className?: string;
  // Runs before the dialog opens, e.g. to close a modal sheet that would
  // otherwise block clicks on Cookiebot's dialog.
  beforeOpen?: () => void | Promise<void>;
}

export function CookieSettingsLink({ className, beforeOpen }: Props) {
  const status = useConsentManagerStatus();

  if (!isConsentManagerConfigured() || status === "unavailable") return null;

  function handleClick() {
    if (!beforeOpen) return openConsentSettings();
    void Promise.resolve(beforeOpen()).then(openConsentSettings);
  }

  return (
    <button
      type="button"
      disabled={status !== "ready"}
      onClick={handleClick}
      className={cn(
        "group inline-flex items-center rounded-md px-2 py-1.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 disabled:cursor-default disabled:opacity-60 sm:py-1",
        className,
      )}
    >
      <Text
        variant="small"
        as="span"
        tone="secondary"
        className="underline-offset-2 group-hover:text-zinc-900 group-hover:underline group-disabled:no-underline"
      >
        Cookie settings
      </Text>
    </button>
  );
}
