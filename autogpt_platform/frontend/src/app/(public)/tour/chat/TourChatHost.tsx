"use client";

import { TourChatContainer } from "./components/TourChatContainer/TourChatContainer";
import type { TourScript } from "./script/types";
import { useTourCopilot } from "./useTourCopilot";

interface Props {
  sessionId: string;
  script: TourScript;
  onComplete?: () => void;
  onReset?: () => void;
}

export function TourChatHost({
  sessionId,
  script,
  onComplete,
  onReset,
}: Props) {
  const chat = useTourCopilot({
    sessionId,
    script,
    onComplete: onComplete ?? (() => {}),
    onReset,
  });

  return <TourChatContainer chat={chat} />;
}
