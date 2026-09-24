"use client";

import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { Flag, useFlagStatus } from "./use-get-flag";

interface Props {
  flag: Flag;
  whenEnabled: string;
  whenDisabled: string;
}

export function FeatureFlagRedirect({
  flag,
  whenEnabled,
  whenDisabled,
}: Props) {
  const router = useRouter();
  const { enabled, ready } = useFlagStatus(flag);
  const flagEnabled = Boolean(enabled);

  useEffect(() => {
    if (!ready) return;
    router.replace(flagEnabled ? whenEnabled : whenDisabled);
  }, [ready, flagEnabled, router, whenEnabled, whenDisabled]);

  return <LoadingSpinner size="large" cover />;
}
