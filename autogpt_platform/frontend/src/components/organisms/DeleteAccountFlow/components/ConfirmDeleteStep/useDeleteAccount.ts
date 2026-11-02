import { usePostAccountDeleteAccount } from "@/app/api/__generated__/endpoints/account/account";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useRouter } from "next/navigation";
import { useState } from "react";
import type { DeletionSource } from "../../helpers";
import { getClosedToastTitle, getDeletionErrorMessage } from "./helpers";

interface Args {
  source: DeletionSource;
  onBusyChange: (busy: boolean) => void;
}

export function useDeleteAccount({ source, onBusyChange }: Args) {
  const router = useRouter();
  const { user } = useAuth();
  const [isLeaving, setIsLeaving] = useState(false);
  const { mutateAsync, isPending } = usePostAccountDeleteAccount();

  async function deleteAccount(confirmEmail: string) {
    onBusyChange(true);
    let eraseAfter: Date | string | null | undefined;
    try {
      const response = await mutateAsync({
        data: { confirm_email: confirmEmail, source },
      });
      if (response.status === 200) eraseAfter = response.data.erase_after;
    } catch (error) {
      onBusyChange(false);
      toast({ title: getDeletionErrorMessage(error), variant: "destructive" });
      return;
    }
    setIsLeaving(true);
    toast({ title: getClosedToastTitle(eraseAfter) });
    // Signing out here would clear the user while the page behind still
    // renders, and it would drop this dialog (Settings turns into a skeleton,
    // the paywall lets go) before the redirect. /logout signs out, then
    // goes to /login.
    router.replace("/logout");
  }

  return {
    email: user?.email ?? "",
    isDeleting: isPending || isLeaving,
    deleteAccount,
  };
}
