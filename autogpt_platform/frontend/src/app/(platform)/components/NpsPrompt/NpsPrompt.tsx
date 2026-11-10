"use client";

import { NpsCard } from "@/components/organisms/FeedbackPrompts/NpsCard";
import { useFeedbackPrompt } from "@/components/organisms/FeedbackPrompts/useFeedbackPrompt";
import { useMountEffect } from "@/hooks/useMountEffect";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useState } from "react";
import {
  isNpsDue,
  markNpsAsked,
  markNpsNotYet,
  shouldCheckNps,
} from "./helpers";

export function NpsPrompt() {
  const { user } = useAuth();
  if (!user) return null;
  return <NpsGate key={user.id} userID={user.id} createdAt={user.created_at} />;
}

interface GateProps {
  userID: string;
  createdAt?: string;
}

// Decides once per account and page load.
function NpsGate({ userID, createdAt }: GateProps) {
  const [isDue] = useState(() => {
    const now = Date.now();
    return isNpsDue(createdAt, now) && shouldCheckNps(userID, now);
  });
  if (!isDue) return null;
  return <NpsClaim userID={userID} />;
}

interface ClaimProps {
  userID: string;
}

function NpsClaim({ userID }: ClaimProps) {
  const prompt = useFeedbackPrompt("nps");
  const [isClaimed, setIsClaimed] = useState(false);

  useMountEffect(() => {
    void prompt.claim().then(({ id, failed }) => {
      if (failed) return;
      if (!id) {
        markNpsNotYet(userID, Date.now());
        return;
      }
      markNpsAsked(userID);
      setIsClaimed(true);
    });
  });

  if (!isClaimed) return null;

  return (
    <div className="fixed right-6 bottom-6 z-40">
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
