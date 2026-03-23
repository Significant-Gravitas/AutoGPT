"use client";

import { useEffect } from "react";
import { WarningCircle } from "@phosphor-icons/react";
import { Button } from "@/components/atoms/Button/Button";

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
          <WarningCircle className="size-10 text-destructive" weight="thin" />
        </div>
        <h1 className="mt-8 text-2xl font-bold tracking-tight text-foreground">
          Oops, something went wrong!
        </h1>
        <p className="mt-4 text-muted-foreground">
          We&apos;re sorry, but an unexpected error has occurred. Please try
          again later or contact support if the issue persists.
        </p>
        <div className="mt-6 flex flex-row justify-center gap-4">
          <Button onClick={reset} variant="ghost">
            Retry
          </Button>
          {/* Full reload to / clears the error boundary state cleanly */}
          <Button onClick={() => (window.location.href = "/")}>
            Go to Homepage
          </Button>
        </div>
      </div>
    </div>
  );
}
