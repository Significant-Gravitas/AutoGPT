"use client";

import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { Suspense } from "react";
import { useErrorPage } from "./useErrorPage";

function ErrorPageContent() {
  const { errorDetails, hideSessionError, handleRetry } = useErrorPage();

  if (hideSessionError) {
    return null;
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50 px-4 py-12 sm:px-6 lg:px-8">
      <div className="relative w-full max-w-xl">
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
          <div className="relative w-full max-w-xl">
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
