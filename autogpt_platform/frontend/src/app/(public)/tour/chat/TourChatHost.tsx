"use client";

import { TourChatContainer } from "./components/TourChatContainer/TourChatContainer";
import type { TourScript } from "./script/types";
import { useTourCopilot } from "./useTourCopilot";

interface Props {
  sessionId: string;
  script: TourScript;
}

export function TourChatHost({ sessionId, script }: Props) {
  const chat = useTourCopilot({
    sessionId,
    script,
    onComplete: () => {},
  });

  return <TourChatContainer chat={chat} />;
}
