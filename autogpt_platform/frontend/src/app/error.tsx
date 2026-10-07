"use client";

import { useEffect } from "react";
import { AlertCircleIcon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="fixed inset-0 flex items-center justify-center bg-background">
      <div className="w-full max-w-md px-4 text-center sm:px-6">
        <div className="mx-auto flex size-12 items-center justify-center rounded-full bg-muted">
          <Icon icon={AlertCircleIcon} className="size-10" />
        </div>
        <Text variant="h4" as="h1" tone="primary" className="mt-8">
          Oops, something went wrong!
        </Text>
        <Text variant="large" tone="muted" className="mt-4">
          We&apos;re sorry, but an unexpected error has occurred. Please try
          again later or contact support if the issue persists.
        </Text>
        <div className="mt-6 flex flex-row justify-center gap-4">
          <Button onClick={reset} variant="outline" size="small">
            Retry
          </Button>
          <Button as="NextLink" href="/" size="small">
            Go to Homepage
          </Button>
        </div>
      </div>
    </div>
  );
}
