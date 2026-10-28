"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { MicOff01Icon } from "@hugeicons/core-free-icons";
import type { MicrophonePermission } from "../useMicrophonePermission";

interface Props {
  isBlocked: boolean;
  // No microphone, or one another app holds: fixed on the device, not in
  // the browser.
  isUnavailable?: boolean;
  permission: MicrophonePermission;
  onRetry: () => void;
}

const COPY = {
  blocked: {
    reason: "Your browser blocked the microphone, so write here instead.",
    howTo:
      "To talk, allow the microphone in your browser's site settings (the lock or microphone icon by the address bar), then",
  },
  // The prompt was closed rather than answered: asking again shows it again.
  dismissed: {
    reason: "The microphone prompt was closed, so write here instead.",
    howTo: "To talk,",
  },
  unavailable: {
    reason: "We couldn't open your microphone, so write here instead.",
    howTo:
      "To talk, check that it's connected and no other app is using it, then",
  },
  // The site is allowed but the operating system refused the browser, as
  // Chrome on a Mac without microphone access in System Settings does.
  system: {
    reason: "Your computer blocked the microphone, so write here instead.",
    howTo:
      "To talk, let your browser use the microphone in your computer's privacy settings, then",
  },
} as const;

export function MicBlockedNotice({
  isBlocked,
  isUnavailable = false,
  permission,
  onRetry,
}: Props) {
  if (!isBlocked && !isUnavailable) return null;
  const copy = isBlocked ? noticeCopy(permission) : COPY.unavailable;

  return (
    <div
      role="alert"
      className="flex w-full max-w-[640px] items-start gap-2.5 self-start rounded-xl bg-surface-hover px-3 py-2.5 text-left"
    >
      <Icon
        icon={MicOff01Icon}
        size={18}
        className="mt-0.5 shrink-0 text-warning-foreground"
      />
      <Text variant="body" tone="secondary">
        <Text variant="body-medium" as="span">
          {copy.reason}
        </Text>{" "}
        {copy.howTo}{" "}
        <Button
          type="button"
          variant="ghost"
          size="xs"
          onClick={onRetry}
          className="inline h-auto rounded-sm p-0 align-baseline text-body text-foreground-base underline underline-offset-4 hover:bg-transparent hover:text-foreground-strong"
        >
          try the microphone again
        </Button>
        .
      </Text>
    </div>
  );
}

function noticeCopy(permission: MicrophonePermission) {
  if (permission === "prompt") return COPY.dismissed;
  if (permission === "granted") return COPY.system;
  return COPY.blocked;
}
