"use client";

import { isServerSide } from "@/lib/utils/is-server-side";
import { useEffect, useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Card } from "@/components/atoms/Card/Card";
import { useRouter } from "next/navigation";

export default function AuthErrorPage() {
  const [errorType, setErrorType] = useState<string | null>(null);
  const [errorCode, setErrorCode] = useState<string | null>(null);
  const [errorDescription, setErrorDescription] = useState<string | null>(null);
  const router = useRouter();

  useEffect(() => {
    // This code only runs on the client side
    if (!isServerSide()) {
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
  const isWaitlistError =
    errorCode === "403" ||
    errorDescription?.toLowerCase().includes("not allowed") ||
    errorDescription?.toLowerCase().includes("waitlist") ||
    errorDescription?.toLowerCase().includes("allowlist");

  if (isWaitlistError) {
    return (
      <div className="flex h-screen items-center justify-center">
        <Card className="w-full max-w-md p-8">
          <div className="flex flex-col items-center gap-6">
            <Text variant="h3">Join the Waitlist</Text>
            <div className="flex flex-col gap-4 text-center">
              <Text variant="body">
                AutoGPT Platform is currently in closed beta. Your email address
                isn&apos;t on our current allowlist for early access.
              </Text>
              <Text variant="small" className="text-muted-foreground">
                Join our waitlist to get notified when we open up access!
              </Text>
            </div>
            <div className="flex gap-3">
              <Button
                variant="secondary"
                onClick={() => {
                  window.open("https://agpt.co/waitlist", "_blank");
                }}
              >
                Join Waitlist
              </Button>
              <Button variant="primary" onClick={() => router.push("/login")}>
                Back to Login
              </Button>
            </div>
            <div className="flex flex-col gap-2">
              <Text
                variant="small"
                className="text-center text-muted-foreground"
              >
                Already signed up for the waitlist? Make sure you&apos;re using
                the exact same email address you used when signing up.
              </Text>
              <Text
                variant="small"
                className="text-center text-muted-foreground"
              >
                If you&apos;re not sure which email you used or need help,{" "}
                <a
                  href="https://discord.gg/autogpt"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline hover:text-foreground"
                >
                  reach out on Discord
                </a>
              </Text>
            </div>
          </div>
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
