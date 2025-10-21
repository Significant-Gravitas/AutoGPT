"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Card } from "@/components/atoms/Card/Card";
import { WaitlistErrorContent } from "@/components/auth/WaitlistErrorContent";
import { isWaitlistErrorFromParams } from "@/app/api/auth/utils";
import { useRouter } from "next/navigation";
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
  const isWaitlistError = isWaitlistErrorFromParams(
    errorCode,
    errorDescription,
  );

  if (isWaitlistError) {
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

  // Default error display for other types of errors
  return (
    <div className="flex h-screen items-center justify-center">
      <Card className="w-full max-w-md p-8">
        <div className="flex flex-col items-center gap-6">
          <Text variant="h3">Authentication Error</Text>
          <div className="flex flex-col gap-2 text-center">
            {errorType && (
              <Text variant="body">
                <strong>Error Type:</strong> {errorType}
              </Text>
            )}
            {errorCode && (
              <Text variant="body">
                <strong>Error Code:</strong> {errorCode}
              </Text>
            )}
            {errorDescription && (
              <Text variant="body">
                <strong>Description:</strong> {errorDescription}
              </Text>
            )}
          </div>
          <Button variant="primary" onClick={() => router.push("/login")}>
            Back to Login
          </Button>
        </div>
      </Card>
    </div>
  );
}
