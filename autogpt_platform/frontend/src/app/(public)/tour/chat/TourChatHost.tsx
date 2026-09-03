"use client";

import { TourChatContainer } from "./components/TourChatContainer/TourChatContainer";
import type { TourScript } from "./script/types";
import { useTourCopilot } from "./useTourCopilot";

interface Props {
  sessionId: string;
  script: TourScript;
  onComplete?: () => void;
  completionNotice?: string;
}

export function TourChatHost({
  sessionId,
  script,
  onComplete,
  completionNotice,
}: Props) {
  const chat = useTourCopilot({
    sessionId,
    script,
    onComplete: onComplete ?? (() => {}),
    completionNotice,
  });

  return <TourChatContainer chat={chat} />;
}
