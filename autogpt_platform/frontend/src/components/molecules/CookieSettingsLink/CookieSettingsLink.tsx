"use client";

import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";
import {
  isConsentManagerConfigured,
  openConsentSettings,
} from "@/services/consent/consent";
import { useConsentManagerStatus } from "@/services/consent/useConsent";

interface Props {
  className?: string;
}

export function CookieSettingsLink({ className }: Props) {
  const status = useConsentManagerStatus();

  if (!isConsentManagerConfigured() || status === "unavailable") return null;

  return (
    <Button
      type="button"
      variant="link"
      disabled={status !== "ready"}
      onClick={openConsentSettings}
      className={cn(
        "text-xs font-normal leading-5 text-zinc-500 no-underline hover:text-zinc-800 hover:underline",
        className,
      )}
    >
      Cookie settings
    </Button>
  );
}
