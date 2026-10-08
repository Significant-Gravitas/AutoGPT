import { useRef, useState } from "react";
import type { ActivationResponse } from "@/app/api/__generated__/models/activationResponse";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import {
  confirmActivation,
  previewActivation,
  retrieveActivation,
  statusCode,
} from "./api";
import { useActivationReadiness } from "./useActivationReadiness";
import { pendingConsent, storeConsent } from "./storage";
import { safeReturnTo } from "./helpers";
import { useActivationRecovery } from "./useActivationRecovery";

export function useActivationController() {
  const userID = useAuthStore((state) => state.user?.id);
  const [attempt, setAttempt] = useState<ActivationResponse | null>(null);
  const [isOpen, setOpen] = useState(false);
  const [isBusy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [needsNewTerms, setNeedsNewTerms] = useState(false);
  const inFlight = useRef(false);
  const checkoutReturn = useRef(
    typeof window !== "undefined" &&
      new URLSearchParams(window.location.search).get("subscription") ===
        "success",
  );
  const interacted = useRef(false);
  const latestAttempt = useRef<ActivationResponse | null>(null);
  const consent = useRef<{ id: string; token: string } | null>(null);
  const destination = useRef("/settings/billing");
  const returnFocus = useRef<HTMLElement | null>(null);
  const [retryConsent, setRetryConsent] = useState(false);
  const { isReady, isRefreshing, finish } = useActivationReadiness(
    userID,
    setOpen,
    setError,
  );

  function isCurrentUser() {
    return !!userID && useAuthStore.getState().user?.id === userID;
  }
  function remember() {
    if (userID && consent.current) storeConsent(userID, consent.current);
    setRetryConsent(true);
  }
  function accept(
    response: ActivationResponse,
    recovered = false,
    background = false,
  ) {
    if (!isCurrentUser() || (recovered && interacted.current)) return;
    if (background && latestAttempt.current?.status === "ready") return;
    const resumed = !!pendingConsent(userID) || checkoutReturn.current;
    if (recovered && !resumed) return;
    latestAttempt.current = response;
    setAttempt(response);
    setError("");
    if (response.status === "ready") {
      void finish(response, !recovered || resumed);
    } else if (
      response.status !== "confirmation_required" ||
      !recovered ||
      resumed
    ) {
      consent.current ??= pendingConsent(userID);
      setRetryConsent(!!consent.current);
      if (!background) setOpen(true);
    }
  }

  function changeOpen(open: boolean) {
    setOpen(open);
    if (!open)
      requestAnimationFrame(() => {
        if (returnFocus.current?.isConnected) returnFocus.current.focus();
      });
  }

  async function start(
    returnTo = `${window.location.pathname}${window.location.search}${window.location.hash}`,
  ) {
    if (inFlight.current || !userID) return;
    interacted.current = true;
    if (!isOpen && document.activeElement instanceof HTMLElement)
      returnFocus.current = document.activeElement;
    setOpen(true);
    setError("");
    if (
      attempt &&
      ["processing", "payment_required", "action_required"].includes(
        attempt.status,
      )
    )
      return;
    const saved = consent.current ?? pendingConsent(userID);
    if (saved) {
      consent.current = saved;
      await check(saved.id);
      return;
    }
    destination.current = safeReturnTo(returnTo);
    inFlight.current = true;
    setBusy(true);
    try {
      let existing: ActivationResponse | null = null;
      try {
        existing = await retrieveActivation();
      } catch (error) {
        if (statusCode(error) !== 404) throw error;
      }
      if (
        existing &&
        existing.status !== "confirmation_required" &&
        existing.status !== "failed"
      )
        accept(existing);
      else {
        const quote = await previewActivation(destination.current);
        consent.current = null;
        setRetryConsent(false);
        setNeedsNewTerms(false);
        accept(quote);
      }
    } catch {
      setError(
        "We couldn’t load your upgrade details. Please try again before confirming payment.",
      );
    } finally {
      inFlight.current = false;
      setBusy(false);
    }
  }

  async function confirm() {
    if (
      inFlight.current ||
      !attempt?.id ||
      !attempt.terms_token ||
      !attempt.terms
    )
      return;
    interacted.current = true;
    inFlight.current = true;
    setBusy(true);
    setError("");
    consent.current ??= { id: attempt.id, token: attempt.terms_token };
    remember();
    try {
      accept(
        await confirmActivation(consent.current.id, consent.current.token),
      );
    } catch (error) {
      try {
        const current = await retrieveActivation(consent.current.id);
        accept(current);
        if (
          current.status === "confirmation_required" &&
          statusCode(error) === 409
        ) {
          consent.current = null;
          if (userID) storeConsent(userID, null);
          setRetryConsent(false);
          setNeedsNewTerms(true);
          setError("Review the current terms before confirming this upgrade.");
        }
      } catch {
        setError(
          "We couldn’t confirm the payment status. Check the existing payment before trying again.",
        );
      }
    } finally {
      inFlight.current = false;
      setBusy(false);
    }
  }

  async function check(id = attempt?.id) {
    if (inFlight.current) return;
    inFlight.current = true;
    setBusy(true);
    try {
      accept(await retrieveActivation(id));
    } catch {
      setError("We couldn’t check your payment yet. Please try again.");
    } finally {
      inFlight.current = false;
      setBusy(false);
    }
  }

  useActivationRecovery({ userID, attempt, accept, onError: setError });
  return {
    attempt,
    isOpen,
    setOpen: changeOpen,
    isBusy: isBusy || isRefreshing,
    isReady,
    error,
    needsNewTerms,
    retryConsent,
    start,
    confirm,
    check,
  };
}
