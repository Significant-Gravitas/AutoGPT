"use client";

import { XIcon } from "@phosphor-icons/react";

import { Button } from "@/components/atoms/Button/Button";
import { Alert, AlertDescription } from "@/components/molecules/Alert/Alert";

import { useLowCreditBanner } from "./useLowCreditBanner";

interface Props {
  className?: string;
}

export function LowCreditBanner({ className }: Props) {
  const { visible, openTopUp, dismiss } = useLowCreditBanner();

  if (!visible) return null;

  const alert = (
    <Alert variant="warning" aria-live="polite">
      <div className="flex flex-wrap items-center gap-3">
        <AlertDescription className="min-w-[12rem] flex-1">
          You&apos;re out of automation credits. Top up to keep your agents
          running.
        </AlertDescription>
        <Button variant="primary" size="small" onClick={openTopUp}>
          Top up
        </Button>
        <Button
          variant="ghost"
          size="icon"
          onClick={dismiss}
          aria-label="Dismiss"
          title="Dismiss"
          className="hover:border-[#FFE4BF] hover:bg-[#FFE4BF]"
        >
          <XIcon className="h-4 w-4" />
        </Button>
      </div>
    </Alert>
  );

  return className ? <div className={className}>{alert}</div> : alert;
}
