"use client";

import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { getErrorDetails } from "./helpers";

function ErrorPageContent() {
  const searchParams = useSearchParams();
  const errorMessage = searchParams.get("message");
  const errorDetails = getErrorDetails(errorMessage);

  function handleRetry() {
    // Auth-related errors should redirect to login
    if (
      errorMessage === "user-creation-failed" ||
      errorMessage === "auth-failed" ||
      errorMessage === "auth-token-invalid" ||
      errorMessage === "session-expired"
    ) {
      window.location.href = "/login";
    } else if (errorMessage === "rate-limited") {
      // For rate limiting, wait a moment then try again
      setTimeout(() => {
        window.location.reload();
      }, 2000);
    } else {
      // For server/network errors, go to marketplace
      window.location.href = "/marketplace";
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50 px-4 py-12 sm:px-6 lg:px-8">
      <div className="relative w-full max-w-xl lg:bottom-[4rem]">
        <ErrorCard
          responseError={errorDetails.responseError}
          context={errorDetails.context}
          onRetry={handleRetry}
        />
      </div>
    </div>
  );
}

export default function ErrorPage() {
  return (
    <Suspense
      fallback={
        <div className="flex min-h-screen items-center justify-center bg-gray-50 px-4 py-12 sm:px-6 lg:px-8">
          <div className="relative w-full max-w-xl lg:-top-[4rem]">
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
