"use client";

import { useEffect, useState } from "react";
import { Text } from "@/components/atoms/Text/Text";
import { Card } from "@/components/atoms/Card/Card";
import { WaitlistErrorContent } from "@/components/auth/WaitlistErrorContent";
import { isWaitlistError } from "@/app/api/auth/utils";
import { useRouter } from "next/navigation";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { environment } from "@/services/environment";

export default function AuthErrorPage() {
  const [errorType, setErrorType] = useState<string | null>(null);
  const [errorCode, setErrorCode] = useState<string | null>(null);
  const [errorDescription, setErrorDescription] = useState<string | null>(null);
  const router = useRouter();

  useEffect(() => {
    // This code only runs on the client side
    if (!environment.isServerSide()) {
      const hash = window.location.hash.substring(1); // Remove the leading '#'
      const params = new URLSearchParams(hash);

      setErrorType(params.get("error"));
      setErrorCode(params.get("error_code"));
      setErrorDescription(
        params.get("error_description")?.replace(/\+/g, " ") ?? null,
      ); // Replace '+' with space
    }
  }, []);

  if (!errorType && !errorCode && !errorDescription) {
    return (
      <div className="flex h-screen items-center justify-center">
        <Text variant="body">Loading...</Text>
      </div>
    );
  }

  // Check if this is a waitlist/not allowed error
  const isWaitlistErr = isWaitlistError(errorCode, errorDescription);

  if (isWaitlistErr) {
    return (
      <div className="flex h-screen items-center justify-center">
        <Card className="w-full max-w-md p-8">
          <WaitlistErrorContent
            onClose={() => router.push("/login")}
            closeButtonText="Back to Login"
          />
        </Card>
      </div>
    );
  }

  // Use ErrorCard for consistent error display
  const errorMessage = errorDescription
    ? `${errorDescription}. If this error persists, please contact support at contact@agpt.co`
    : "An authentication error occurred. Please contact support at contact@agpt.co";

  return (
    <div className="flex h-screen items-center justify-center p-4">
      <div className="w-full max-w-md">
        <ErrorCard
          responseError={{
            message: errorMessage,
            detail: errorCode
              ? `Error code: ${errorCode}${errorType ? ` (${errorType})` : ""}`
              : undefined,
          }}
          context="authentication"
          onRetry={() => router.push("/login")}
        />
      </div>
    </div>
  );
}
