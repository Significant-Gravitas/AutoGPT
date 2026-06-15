"use client";

import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect } from "react";
import { getErrorDetails } from "./helpers";

function ErrorPageContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const { isLoggedIn, isUserLoading } = useSupabase();
  const errorMessage = searchParams.get("message");
  const errorDetails = getErrorDetails(errorMessage);
  const shouldRedirectLoggedOutSessionExpired =
    errorMessage === "session-expired" && !isUserLoading && !isLoggedIn;

  useEffect(() => {
    if (shouldRedirectLoggedOutSessionExpired) {
      router.replace("/login");
    }
  }, [router, shouldRedirectLoggedOutSessionExpired]);

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
      // For server/network errors, go to home
      window.location.href = "/";
    }
  }

  if (shouldRedirectLoggedOutSessionExpired) {
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
