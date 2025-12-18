"use client";

import * as Sentry from "@sentry/nextjs";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { useEffect } from "react";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    Sentry.captureException(error);
  }, [error]);

  return (
    <html>
      <body>
        <div className="flex min-h-screen items-center justify-center bg-gray-50 px-4 py-12 sm:px-6 lg:px-8">
          <div className="relative w-full max-w-xl lg:bottom-[4rem]">
            <ErrorCard
              responseError={{
                message:
                  error.message ||
                  "An unexpected error occurred. Our team has been notified and is working to resolve the issue.",
              }}
              context="application"
              onRetry={reset}
            />
          </div>
        </div>
      </body>
    </html>
  );
}
