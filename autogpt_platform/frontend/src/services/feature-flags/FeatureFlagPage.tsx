"use client";

import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { useRouter } from "next/navigation";
import { ReactNode, useEffect } from "react";
import { Flag, useFlagStatus } from "./use-get-flag";

interface Props {
  flag: Flag;
  whenDisabled: string;
  children: ReactNode;
}

export function FeatureFlagPage({ flag, whenDisabled, children }: Props) {
  const router = useRouter();
  const { enabled, ready } = useFlagStatus(flag);
  const flagEnabled = Boolean(enabled);

  useEffect(() => {
    if (ready && !flagEnabled) {
      router.replace(whenDisabled);
    }
  }, [ready, flagEnabled, router, whenDisabled]);

  return !ready || !flagEnabled ? (
    <LoadingSpinner size="large" cover />
  ) : (
    <>{children}</>
  );
}
