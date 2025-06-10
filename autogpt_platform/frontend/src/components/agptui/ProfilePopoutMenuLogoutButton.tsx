"use client";
import useSupabase from "@/lib/supabase/useSupabase";
import { IconLogOut } from "@/components/ui/icons";
import { useState } from "react";
import { LoadingSpinner } from "../ui/loading";
import { cn } from "@/lib/utils";
import { useRouter } from "next/navigation";
import { toast } from "../ui/use-toast";

export function ProfilePopoutMenuLogoutButton() {
  const router = useRouter();
  const [isLoggingOut, setIsLoggingOut] = useState(false);
  const supabase = useSupabase();

  async function handleLogout() {
    setIsLoggingOut(true);

    try {
      await supabase.logOut();
      router.refresh();
    } catch (e) {
      console.error(e);
      toast({
        title: "Error logging out",
        description:
          "Something went wrong when logging out. Please try again. If the problem persists, please contact support.",
        variant: "destructive",
      });
      setIsLoggingOut(false);
    }
  }

  return (
    <div
      className={cn(
        "inline-flex w-full items-center justify-start gap-2.5",
        isLoggingOut && "justify-center",
      )}
      onClick={handleLogout}
      role="button"
      tabIndex={0}
    >
      {isLoggingOut ? (
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
