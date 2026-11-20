import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect } from "react";
import { getErrorDetails } from "./helpers";

export function useErrorPage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const { isLoggedIn, isUserLoading } = useAuth();
  const errorMessage = searchParams.get("message");
  const expiredSession = errorMessage === "session-expired";
  const redirectToLogin = expiredSession && !isUserLoading && !isLoggedIn;

  useEffect(() => {
    if (redirectToLogin) router.replace("/login");
  }, [router, redirectToLogin]);

  function handleRetry() {
    if (
      errorMessage === "user-creation-failed" ||
      errorMessage === "auth-failed" ||
      errorMessage === "auth-token-invalid" ||
      expiredSession
    ) {
      window.location.replace("/login");
    } else if (errorMessage === "rate-limited") {
      setTimeout(() => window.location.reload(), 2000);
    } else {
      router.replace("/");
    }
  }

  return {
    errorDetails: getErrorDetails(errorMessage),
    hideSessionError: expiredSession && (isUserLoading || !isLoggedIn),
    handleRetry,
  };
}
