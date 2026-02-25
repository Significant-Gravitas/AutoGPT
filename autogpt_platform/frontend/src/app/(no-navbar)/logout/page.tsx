"use client";

import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useRouter } from "next/navigation";
import { useEffect, useRef } from "react";

const LOGOUT_REDIRECT_DELAY_MS = 400;

function wait(ms: number): Promise<void> {
  return new Promise(function resolveAfterDelay(resolve) {
    setTimeout(resolve, ms);
  });
}

export default function LogoutPage() {
  const { logOut } = useSupabase();
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
          toast({
            title: "Failed to log out. Redirecting to login.",
            variant: "destructive",
          });
        } finally {
          await wait(LOGOUT_REDIRECT_DELAY_MS);
          router.replace("/login");
        }
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
