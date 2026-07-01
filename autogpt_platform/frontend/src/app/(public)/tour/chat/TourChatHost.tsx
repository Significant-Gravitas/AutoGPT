"use client";

import { useState } from "react";
import { TourChatContainer } from "./components/TourChatContainer/TourChatContainer";
import { TourUpsellModal } from "./components/TourUpsellModal/TourUpsellModal";
import type { TourScript } from "./script/types";
import { useTourCopilot } from "./useTourCopilot";

interface Props {
  sessionId: string;
  script: TourScript;
}

export function TourChatHost({ sessionId, script }: Props) {
  const [isUpsellOpen, setIsUpsellOpen] = useState(false);
  const chat = useTourCopilot({
    sessionId,
    script,
    onComplete: () => setIsUpsellOpen(true),
  });

  return (
    <>
      <TourChatContainer chat={chat} />
      <TourUpsellModal
        open={isUpsellOpen}
        onClose={() => setIsUpsellOpen(false)}
        onReplay={() => {
          chat.reset();
          setIsUpsellOpen(false);
        }}
      />
    </>
  );
}
