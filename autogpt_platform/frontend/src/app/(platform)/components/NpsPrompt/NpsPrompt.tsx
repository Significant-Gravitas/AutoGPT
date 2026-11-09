"use client";

import { useTopUpPrompt } from "@/components/layout/TopUpPrompt/useTopUpPrompt";
import { NpsCard } from "@/components/organisms/FeedbackPrompts/NpsCard";
import { useFeedbackPrompt } from "@/components/organisms/FeedbackPrompts/useFeedbackPrompt";
import { useMountEffect } from "@/hooks/useMountEffect";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { cn } from "@/lib/utils";
import { useState } from "react";
import { hasNpsMarker, isNpsDue, setNpsMarker } from "./helpers";

export function NpsPrompt() {
  const { user } = useAuth();
  if (!user) return null;
  return <NpsGate key={user.id} userID={user.id} createdAt={user.created_at} />;
}

interface GateProps {
  userID: string;
  createdAt?: string;
}

// Decides once per account and page load; the top-up card only holds the
// question back, it never takes it away once asked.
function NpsGate({ userID, createdAt }: GateProps) {
  const { isTopUpCardOpen } = useTopUpPrompt();
  const [isDue] = useState(
    () => isNpsDue(createdAt, Date.now()) && !hasNpsMarker(userID),
  );
  const [isStarted, setIsStarted] = useState(false);

  if (isDue && !isTopUpCardOpen && !isStarted) setIsStarted(true);
  if (!isStarted) return null;

  return <NpsClaim userID={userID} isHidden={isTopUpCardOpen} />;
}

interface ClaimProps {
  userID: string;
  isHidden: boolean;
}

function NpsClaim({ userID, isHidden }: ClaimProps) {
  const prompt = useFeedbackPrompt("nps");
  const [isClaimed, setIsClaimed] = useState(false);

  useMountEffect(() => {
    void prompt.claim().then(({ id, failed }) => {
      if (!failed) setNpsMarker(userID);
      if (id) setIsClaimed(true);
    });
  });

  if (!isClaimed) return null;

  return (
    <div className={cn("fixed right-6 bottom-6 z-40", isHidden && "hidden")}>
      <NpsCard
        onScore={(score) => prompt.save({ score })}
        onSubmit={({ score, note }) =>
          prompt.save({ score, note: note || undefined })
        }
        onDismiss={() => prompt.save({ dismissed: true })}
      />
    </div>
  );
}
