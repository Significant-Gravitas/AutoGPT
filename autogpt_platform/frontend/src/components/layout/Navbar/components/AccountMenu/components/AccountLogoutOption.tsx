"use client";
import { IconLogOut } from "@/components/__legacy__/ui/icons";
import { LoadingSpinner } from "@/components/__legacy__/ui/loading";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { cn } from "@/lib/utils";
import * as Sentry from "@sentry/nextjs";
import { useTransition } from "react";

export function AccountLogoutOption() {
  const [isPending, startTransition] = useTransition();
  const supabase = useSupabase();

  function handleLogout() {
    startTransition(async () => {
      try {
        await supabase.logOut();
      } catch (e) {
        Sentry.captureException(e);
        toast({
          title: "Error logging out",
          description:
            "Something went wrong when logging out. Please try again. If the problem persists, please contact support.",
          variant: "destructive",
        });
      }
    });
  }

  return (
    <div
      className={cn(
        "inline-flex w-full items-center justify-start gap-2.5",
        isPending && "justify-center",
      )}
      onClick={handleLogout}
      role="button"
      tabIndex={0}
    >
      {isPending ? (
        <LoadingSpinner className="size-5" />
      ) : (
        <>
          <div className="relative h-6 w-6">
            <IconLogOut className="h-6 w-6" />
          </div>
          <div className="font-sans text-base font-medium leading-normal text-neutral-800 dark:text-neutral-200">
            Log out
          </div>
        </>
      )}
    </div>
  );
}
