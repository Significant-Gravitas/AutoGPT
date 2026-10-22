"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";

export function ReportLoading() {
  return (
    <div className="flex flex-col gap-2" aria-busy="true">
      <Skeleton className="h-10 w-full rounded-lg" />
      <Skeleton className="h-10 w-full rounded-lg" />
    </div>
  );
}

interface ErrorProps {
  message: string;
  isForbidden: boolean;
  onRetry: () => void;
}

export function ReportError({ message, isForbidden, onRetry }: ErrorProps) {
  if (isForbidden) {
    return (
      <Text variant="small" className="text-zinc-500">
        You no longer have admin access. Sign in again.
      </Text>
    );
  }
  return (
    <div className="flex items-center justify-between gap-4">
      <Text variant="small" className="text-zinc-500">
        {message}
      </Text>
      <Button variant="secondary" size="small" onClick={onRetry}>
        Retry
      </Button>
    </div>
  );
}
