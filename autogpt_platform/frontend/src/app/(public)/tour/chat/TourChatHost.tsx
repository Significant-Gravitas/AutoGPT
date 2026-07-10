"use client";

import { TourChatContainer } from "./components/TourChatContainer/TourChatContainer";
import type { TourScript } from "./script/types";
import { useTourCopilot } from "./useTourCopilot";

interface Props {
  sessionId: string;
  script: TourScript;
  onComplete?: () => void;
}

export function TourChatHost({ sessionId, script, onComplete }: Props) {
  const chat = useTourCopilot({
    sessionId,
    script,
    onComplete: onComplete ?? (() => {}),
  });

  return <TourChatContainer chat={chat} />;
}
