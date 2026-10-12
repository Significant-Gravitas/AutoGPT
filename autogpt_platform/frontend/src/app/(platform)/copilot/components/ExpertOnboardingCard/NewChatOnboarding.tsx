"use client";

import { useGetExpertOnboarding } from "@/app/api/__generated__/endpoints/experts/experts";
import type { ReactNode } from "react";
import { ExpertOnboardingCard } from "./ExpertOnboardingCard";
import { PendingOnboardingContext } from "./PendingOnboardingContext";

interface Props {
  expertId: string | null;
  enabled?: boolean;
  children?: ReactNode;
}

export function NewChatOnboarding({
  expertId,
  enabled = true,
  children,
}: Props) {
  const query = useGetExpertOnboarding(expertId ?? "", {
    query: {
      enabled: enabled && !!expertId,
      staleTime: 0,
      refetchOnMount: "always",
    },
  });
  const onboarding = query.data?.status === 200 ? query.data.data : null;
  if (
    !enabled ||
    !expertId ||
    !query.isFetchedAfterMount ||
    query.isError ||
    !onboarding
  ) {
    return children;
  }
  const callId = `pending-onboarding-${expertId}`;

  return (
    <div className="mx-auto min-h-0 w-full max-w-3xl flex-1 overflow-y-auto px-3 py-6 text-left">
      <PendingOnboardingContext.Provider value={callId}>
        <ExpertOnboardingCard
          key={callId}
          part={{
            type: "tool-expert_onboarding",
            toolCallId: callId,
            state: "output-available",
            input: {},
            output: onboarding,
          }}
        />
      </PendingOnboardingContext.Provider>
    </div>
  );
}
