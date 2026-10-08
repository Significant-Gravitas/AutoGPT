"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { Mail01Icon } from "@hugeicons/core-free-icons";
import {
  CHECK_YOUR_INBOX_COPY,
  CHECK_YOUR_INBOX_HEADING_ID,
  type CheckYourInboxReason,
} from "./helpers";
import { useCheckYourInbox } from "./useCheckYourInbox";

interface Props {
  email: string;
  reason: CheckYourInboxReason;
  next?: string | null;
  marketingOptOut?: boolean;
  onBack: () => void;
}

export function CheckYourInbox({
  email,
  reason,
  next,
  marketingOptOut,
  onBack,
}: Props) {
  const copy = CHECK_YOUR_INBOX_COPY[reason];
  const { cooldown, isResending, canResend, handleResend } = useCheckYourInbox({
    email,
    next,
    marketingOptOut,
  });

  return (
    <div className="flex w-full flex-col items-center text-center">
      <div className="mb-6 flex size-14 items-center justify-center rounded-full bg-violet-50">
        <Icon icon={Mail01Icon} className="size-7 text-violet-600" />
      </div>

      <Text
        id={CHECK_YOUR_INBOX_HEADING_ID}
        tabIndex={-1}
        variant="h3"
        as="h1"
        className="!text-slate-950 outline-none"
      >
        {copy.title}
      </Text>

      <Text variant="body" tone="secondary" unmask={false} className="mt-3">
        We sent an email to{" "}
        <span className="break-all font-medium text-slate-950">{email}</span>.
        Open the link in it to {copy.action}.
      </Text>

      <Text variant="small" tone="muted" className="mt-2">
        Can&apos;t find it? Check your spam folder.
      </Text>

      <Button
        variant="secondary"
        loading={isResending}
        type="button"
        disabled={!canResend}
        onClick={handleResend}
        className="mt-8 w-full"
      >
        {cooldown > 0 ? `Resend email in ${cooldown}s` : "Resend email"}
      </Button>

      <div className="mt-6 inline-flex items-center justify-center gap-1">
        <Text variant="body-medium" className="!text-slate-500">
          {copy.backPrompt}
        </Text>
        <Button
          type="button"
          variant="link"
          className="h-auto min-w-0 p-0"
          onClick={onBack}
        >
          {copy.backLabel}
        </Button>
      </div>
    </div>
  );
}
