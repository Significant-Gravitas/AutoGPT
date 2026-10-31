import { useEffect, useRef } from "react";
import { focusManager, onlineManager } from "@tanstack/react-query";
import type { ActivationResponse } from "@/app/api/__generated__/models/activationResponse";
import { retrieveActivation, statusCode } from "./api";
import { pendingConsent } from "./storage";

interface Options {
  userID?: string;
  attempt: ActivationResponse | null;
  /** The page loaded on the way back from Stripe's checkout. */
  checkoutReturn: boolean;
  accept: (
    response: ActivationResponse,
    recovered?: boolean,
    background?: boolean,
  ) => void;
  onError: (message: string) => void;
  /** The server has no activation for the attempt being checked. */
  onMissing: () => void;
}

export function useActivationRecovery(options: Options) {
  const latest = useRef(options);
  latest.current = options;
  const { userID, attempt, checkoutReturn } = options;
  useEffect(() => {
    // Only a payment being resumed can use the answer, and for a paying user
    // the lookup reconciles with Stripe, so an ordinary page load skips it.
    if (!userID || (!checkoutReturn && !pendingConsent(userID))) return;
    let alive = true;
    retrieveActivation()
      .then((result) => {
        if (alive) latest.current.accept(result, true);
      })
      .catch((error) => {
        if (alive && statusCode(error) !== 404) {
          latest.current.onError(
            "We couldn’t check your payment yet. Your activation can be recovered without starting another purchase.",
          );
        }
      });
    return () => {
      alive = false;
    };
  }, [userID, checkoutReturn]);

  useEffect(() => {
    if (
      !userID ||
      !attempt ||
      !["processing", "payment_required", "action_required"].includes(
        attempt.status,
      )
    )
      return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    let inFlight = false;
    function canPoll() {
      return alive && focusManager.isFocused() && onlineManager.isOnline();
    }
    function schedule() {
      clearTimeout(timer);
      if (canPoll())
        timer = setTimeout(
          poll,
          Math.max(3, attempt?.retry_after_seconds ?? 3) * 1000,
        );
    }
    async function poll() {
      if (!canPoll() || inFlight) return;
      inFlight = true;
      try {
        const response = await retrieveActivation(attempt?.id);
        if (alive) latest.current.accept(response, false, true);
      } catch (error) {
        // An attempt the server no longer has is nothing to wait for.
        if (alive && statusCode(error) === 404) latest.current.onMissing();
        else if (alive)
          latest.current.onError(
            "We’re still checking your payment. You won’t be charged again by this status check.",
          );
      } finally {
        inFlight = false;
        schedule();
      }
    }
    function resume() {
      clearTimeout(timer);
      void poll();
    }
    const unsubscribeFocus = focusManager.subscribe(resume);
    const unsubscribeOnline = onlineManager.subscribe(resume);
    schedule();
    return () => {
      alive = false;
      clearTimeout(timer);
      unsubscribeFocus();
      unsubscribeOnline();
    };
  }, [userID, attempt?.id, attempt?.status, attempt?.retry_after_seconds]);
}
