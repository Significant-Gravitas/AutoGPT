"use client";

import { XIcon } from "@phosphor-icons/react";

import { Button } from "@/components/atoms/Button/Button";
import { Alert, AlertDescription } from "@/components/molecules/Alert/Alert";

import { useLowCreditBanner } from "./useLowCreditBanner";

export function LowCreditBanner() {
  const { visible, openTopUp, dismiss } = useLowCreditBanner();

  if (!visible) return null;

  return (
    <Alert variant="warning">
      <div className="flex items-center gap-3">
        <AlertDescription className="flex-1">
          You&apos;re out of automation credits — top up to keep your agents
          running.
        </AlertDescription>
        <Button variant="primary" size="small" onClick={openTopUp}>
          Top up
        </Button>
        <Button
          variant="icon"
          size="icon"
          onClick={dismiss}
          aria-label="Dismiss"
        >
          <XIcon className="h-4 w-4" />
        </Button>
      </div>
    </Alert>
  );
}
