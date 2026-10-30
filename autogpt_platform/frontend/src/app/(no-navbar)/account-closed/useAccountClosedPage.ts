import {
  useGetAccountGetAccountDeletionStatus,
  usePostAccountKeepMyAccount,
} from "@/app/api/__generated__/endpoints/account/account";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { getEraseDateLabel, getKeepErrorMessage } from "./helpers";

export function useAccountClosedPage() {
  const router = useRouter();
  const [isLeaving, setIsLeaving] = useState(false);
  const status = useGetAccountGetAccountDeletionStatus({
    query: {
      select: (response) =>
        response.status === 200 ? response.data : undefined,
    },
  });
  const { mutateAsync, isPending } = usePostAccountKeepMyAccount();
  const isOpen = status.data?.status === "open";

  useEffect(
    function leaveAnOpenAccount() {
      // Nothing to keep: the account was never closed, or was kept already.
      if (isOpen) window.location.replace("/");
    },
    [isOpen],
  );

  async function keepAccount() {
    try {
      await mutateAsync();
    } catch (error) {
      toast({ title: getKeepErrorMessage(error), variant: "destructive" });
      return;
    }
    setIsLeaving(true);
    toast({ title: "Your account is open again." });
    // A full load, so every query starts over as the open account.
    window.location.replace("/");
  }

  function signOut() {
    setIsLeaving(true);
    router.replace("/logout");
  }

  return {
    eraseDate: getEraseDateLabel(status.data?.erase_after),
    isLoading: status.isLoading || isOpen,
    isError: status.isError || (!status.isLoading && !status.data),
    isKeeping: isPending || isLeaving,
    keepAccount,
    signOut,
  };
}
