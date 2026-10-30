import {
  useGetAccountGetAccountDeletionStatus,
  usePostAccountKeepMyAccount,
} from "@/app/api/__generated__/endpoints/account/account";
import { getV1GetUserTimezone } from "@/app/api/__generated__/endpoints/auth/auth";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import {
  OPEN_PROBE,
  getEraseDateLabel,
  getKeepErrorMessage,
  getPageView,
  waitUntilAnsweredAsOpen,
} from "./helpers";

export function useAccountClosedPage() {
  const router = useRouter();
  const [isKept, setIsKept] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [isLeaving, setIsLeaving] = useState(false);
  const status = useGetAccountGetAccountDeletionStatus({
    query: {
      select: (response) =>
        response.status === 200 ? response.data : undefined,
    },
  });
  const { mutateAsync, isPending } = usePostAccountKeepMyAccount();
  const view = getPageView({
    isKept,
    isLoading: status.isLoading,
    isError: status.isError,
    status: status.data?.status,
    eraseAfter: status.data?.erase_after,
  });
  const isOpen = view === "open";

  useEffect(
    function waitUntilOpenEverywhere() {
      // Another server may still answer the account as closed for a few
      // seconds and would send it straight back here, so Continue shows only
      // once ordinary requests stop being refused.
      if (!isOpen) return;
      let cancelled = false;
      waitUntilAnsweredAsOpen({
        probe: () => getV1GetUserTimezone(),
        isCancelled: () => cancelled,
        settings: OPEN_PROBE,
      }).then(function showContinue() {
        if (!cancelled) setIsReady(true);
      });
      return function stopWaiting() {
        cancelled = true;
      };
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
    setIsKept(true);
  }

  // A full load, so every query starts over as the open account.
  function continueToApp() {
    setIsLeaving(true);
    window.location.replace("/");
  }

  function signOut() {
    setIsLeaving(true);
    router.replace("/logout");
  }

  return {
    view,
    eraseDate: getEraseDateLabel(status.data?.erase_after),
    isReady,
    isKeeping: isPending,
    isLeaving,
    keepAccount,
    continueToApp,
    signOut,
  };
}
