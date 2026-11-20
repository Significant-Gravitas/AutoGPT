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
  isTooLateError,
  waitUntilAnsweredAsOpen,
} from "./helpers";

export function useAccountClosedPage() {
  const router = useRouter();
  const [keptAt, setKeptAt] = useState<number | null>(null);
  const [isTooLate, setIsTooLate] = useState(false);
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
    isKept: keptAt !== null,
    isTooLate,
    isLoading: status.isLoading,
    isError: status.isError,
    status: status.data?.status,
  });
  const isLeavingOpen = view === "leaving";

  useEffect(
    function leaveAnOpenAccount() {
      // Never closed, or kept before this visit. Another server may still
      // answer it as closed and send the app straight back here, so this
      // waits for ordinary requests to answer it as open first (or gives up
      // after the probe cap and goes anyway).
      if (!isLeavingOpen) return;
      let cancelled = false;
      waitUntilAnsweredAsOpen({
        probe: () => getV1GetUserTimezone(),
        isCancelled: () => cancelled,
        settings: { ...OPEN_PROBE, minWaitMS: 0 },
        keptAt: 0,
      }).then(function goToTheApp() {
        if (!cancelled) window.location.replace("/");
      });
      return function stopWaiting() {
        cancelled = true;
      };
    },
    [isLeavingOpen],
  );

  useEffect(
    function waitUntilOpenEverywhere() {
      // Every server caches "closed" for up to 30 s and would send the tab
      // straight back here, so Continue shows only once that has run out
      // and ordinary requests answer the account as open.
      if (keptAt === null) return;
      let cancelled = false;
      waitUntilAnsweredAsOpen({
        probe: () => getV1GetUserTimezone(),
        isCancelled: () => cancelled,
        settings: OPEN_PROBE,
        keptAt,
      }).then(function showContinue() {
        if (!cancelled) setIsReady(true);
      });
      return function stopWaiting() {
        cancelled = true;
      };
    },
    [keptAt],
  );

  async function keepAccount() {
    try {
      await mutateAsync();
    } catch (error) {
      if (isTooLateError(error)) {
        setIsTooLate(true);
        return;
      }
      toast({ title: getKeepErrorMessage(error), variant: "destructive" });
      return;
    }
    setKeptAt(Date.now());
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
