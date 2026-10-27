"use client";

import { cn } from "@/lib/utils";
import { useIsShortScreen } from "../useIsShortScreen";
import type { MicrophonePermission } from "../useMicrophonePermission";
import { MicBlockedNotice } from "./MicBlockedNotice";
import { OttoMightAsk } from "./OttoMightAsk";
import { TypedFallback } from "./TypedFallback";

interface Props {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  isTooLong: boolean;
  question: string;
  onTogglePause: () => void;
  announcement: string;
  onAnotherQuestion: () => void;
  isMicBlocked: boolean;
  isMicUnavailable: boolean;
  micPermission: MicrophonePermission;
  onRetryMicrophone: () => void;
}

export function TypingComposer({
  value,
  onChange,
  onSubmit,
  isTooLong,
  question,
  onTogglePause,
  announcement,
  onAnotherQuestion,
  isMicBlocked,
  isMicUnavailable,
  micPermission,
  onRetryMicrophone,
}: Props) {
  const isShortScreen = useIsShortScreen();
  const hasNotice = isMicBlocked || isMicUnavailable;

  return (
    <div
      className={cn("flex w-full flex-col", isShortScreen ? "gap-2" : "gap-3")}
    >
      <MicBlockedNotice
        isBlocked={isMicBlocked}
        isUnavailable={isMicUnavailable}
        permission={micPermission}
        onRetry={onRetryMicrophone}
      />
      <OttoMightAsk
        question={question}
        onAnother={onAnotherQuestion}
        onTogglePause={onTogglePause}
        announcement={announcement}
        align="left"
      />
      {/* Fewer rows on a short laptop screen, and one fewer while the
          notice is up, keep "I'm done" and "record instead" above the fold
          at 1280x720. */}
      <TypedFallback
        value={value}
        onChange={onChange}
        isTooLong={isTooLong}
        onSubmit={onSubmit}
        rows={(isShortScreen ? 4 : 6) - (hasNotice ? 1 : 0)}
      />
    </div>
  );
}
