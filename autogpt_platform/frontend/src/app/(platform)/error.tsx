"use client";

import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { reloadForChunkLoadError } from "@/lib/chunk-load-reload";
import { getErrorDetails } from "./error/helpers";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect } from "react";

interface Props {
  error: Error & { digest?: string };
}

function ErrorPageContent() {
  const searchParams = useSearchParams();
  const errorMessage = searchParams.get("message");

  const errorDetails = getErrorDetails(errorMessage);

  function handleRetry() {
    if (
      errorMessage === "user-creation-failed" ||
      errorMessage === "auth-failed"
    ) {
      window.location.href = "/login";
    } else {
      window.document.location.reload();
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-surface-muted px-4 py-12 sm:px-6 lg:px-8">
      <div className="w-full max-w-md">
        <ErrorCard
          responseError={errorDetails.responseError}
          context={errorDetails.context}
          onRetry={handleRetry}
        />
      </div>
    </div>
  );
}

export default function ErrorPage({ error }: Props) {
  useEffect(() => {
    reloadForChunkLoadError(error, "error_boundary");
  }, [error]);

  return (
    <Suspense
      fallback={
        <div className="flex min-h-screen items-center justify-center bg-surface-muted px-4 py-12 sm:px-6 lg:px-8">
          <div className="w-full max-w-md">
            <ErrorCard
              responseError={{ message: "Loading..." }}
              context="application"
            />
          </div>
        </div>
      }
    >
      <ErrorPageContent />
    </Suspense>
  );
}
