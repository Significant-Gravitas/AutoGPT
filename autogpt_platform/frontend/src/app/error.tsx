"use client";

import { useEffect } from "react";
import { IconCircleAlert } from "@/components/__legacy__/ui/icons";
import { Button } from "@/components/atoms/Button/Button";
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
          <IconCircleAlert className="size-10" />
        </div>
        <Text
          variant="h3"
          as="h1"
          className="mt-8 text-2xl font-bold tracking-tight text-foreground"
        >
          Oops, something went wrong!
        </Text>
        <Text as="p" className="mt-4 text-muted-foreground">
          We&apos;re sorry, but an unexpected error has occurred. Please try
          again later or contact support if the issue persists.
        </Text>
        <div className="mt-6 flex flex-row justify-center gap-4">
          <Button onClick={reset} variant="outline">
            Retry
          </Button>
          <Button as="NextLink" href="/">
            Go to Homepage
          </Button>
        </div>
      </div>
    </div>
  );
}
