"use client";

import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useRouter } from "next/navigation";
import { useEffect, useRef } from "react";

const LOGOUT_REDIRECT_DELAY_MS = 400;

function wait(ms: number): Promise<void> {
  return new Promise(function resolveAfterDelay(resolve) {
    setTimeout(resolve, ms);
  });
}

export default function LogoutPage() {
  const { logOut } = useAuth();
  const { toast } = useToast();
  const router = useRouter();
  const hasStartedRef = useRef(false);

  useEffect(
    function handleLogoutEffect() {
      if (hasStartedRef.current) return;
      hasStartedRef.current = true;

      async function runLogout() {
        try {
          await logOut();
        } catch {
          // The sign-out never reached the server, so the session is still
          // live. Say so, and leave the user in the app rather than on a
          // login page that would only send them back in.
          toast({
            title: "Couldn't sign you out",
            description: "Check your connection and try again.",
            variant: "destructive",
          });
          router.replace("/");
          return;
        }
        await wait(LOGOUT_REDIRECT_DELAY_MS);
        router.replace("/login");
      }

      void runLogout();
    },
    [logOut, router, toast],
  );

  return (
    <div className="flex min-h-screen items-center justify-center px-4">
      <div className="flex flex-col items-center justify-center gap-4 py-8">
        <LoadingSpinner size="large" />
        <Text variant="body" className="text-center">
          Logging you out...
        </Text>
      </div>
    </div>
  );
}
