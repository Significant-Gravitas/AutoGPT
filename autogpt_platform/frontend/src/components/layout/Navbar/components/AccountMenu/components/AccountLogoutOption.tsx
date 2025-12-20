"use client";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useAuth } from "@/lib/auth";
import { cn } from "@/lib/utils";
import { CircleNotch, SignOut } from "@phosphor-icons/react";
import * as Sentry from "@sentry/nextjs";
import { useRouter } from "next/navigation";
import { useState } from "react";

export function AccountLogoutOption() {
  const [isLoggingOut, setIsLoggingOut] = useState(false);
  const { logOut } = useAuth();
  const router = useRouter();
  const { toast } = useToast();

  async function handleLogout() {
    setIsLoggingOut(true);
    try {
      await logOut();
      router.push("/login");
    } catch (e) {
      Sentry.captureException(e);
      toast({
        title: "Error logging out",
        description:
          "Something went wrong when logging out. Please try again. If the problem persists, please contact support.",
        variant: "destructive",
      });
    } finally {
      setTimeout(() => {
        setIsLoggingOut(false);
      }, 3000);
    }
  }

  return (
    <div
      className={cn(
        "inline-flex w-full items-center justify-start gap-2.5",
        isLoggingOut && "justify-center opacity-50",
      )}
      onClick={handleLogout}
      role="button"
      tabIndex={0}
    >
      {isLoggingOut ? (
        <CircleNotch className="size-5 animate-spin" weight="bold" />
      ) : (
        <>
          <div className="relative h-4 w-4">
            <SignOut className="h-4 w-4" />
          </div>
          <div className="font-sans text-base font-medium leading-normal text-neutral-800 dark:text-neutral-200">
            Log out
          </div>
        </>
      )}
    </div>
  );
}
