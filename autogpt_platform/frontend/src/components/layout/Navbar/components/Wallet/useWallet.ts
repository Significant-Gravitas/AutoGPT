"use client";

import { AGPT_CONFETTI_COLORS } from "@/components/molecules/Confetti/Confetti";
import useCredits from "@/hooks/useCredits";
import {
  OnboardingStep,
  WebSocketNotification,
} from "@/lib/autogpt-server-api";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { useOnboarding } from "@/providers/onboarding/onboarding-provider";
import confetti, { type Options as ConfettiOptions } from "canvas-confetti";
import { useEffect, useRef, useState } from "react";
import { getTaskGroups } from "./helpers";

export function useWallet() {
  const { state, updateState } = useOnboarding();
  const api = useBackendAPI();
  const { credits, formatCredits, fetchCredits } = useCredits({
    fetchInitialCredits: true,
  });

  const groups = getTaskGroups(state);

  const [prevCredits, setPrevCredits] = useState<number | null>(credits);
  const [flash, setFlash] = useState(false);
  const [walletOpen, setWalletOpen] = useState(false);
  const [topUpOpen, setTopUpOpen] = useState(false);

  const walletRef = useRef<HTMLButtonElement | null>(null);

  const totalCount = groups.reduce((acc, group) => acc + group.tasks.length, 0);

  // Total completed task count across all groups
  const completedCount = state
    ? groups.reduce(
        (acc, group) =>
          acc +
          group.tasks.filter((task) => state.completedSteps?.includes(task.id))
            .length,
        0,
      )
    : null;

  async function onWalletOpen() {
    if (!state?.walletShown) {
      updateState({ walletShown: true });
    }
    // Refresh credits when the wallet is opened
    fetchCredits();
  }

  function onAddCredits() {
    setWalletOpen(false);
    setTopUpOpen(true);
  }

  function onTopUpClose() {
    setTopUpOpen(false);
  }

  // React to onboarding notifications emitted by the provider
  function handleNotification(notification: WebSocketNotification) {
    if (
      notification.type !== "onboarding" ||
      notification.event !== "step_completed"
    ) {
      return;
    }

    // Always refresh credits when any onboarding step completes
    fetchCredits();

    // Only trigger confetti for tasks that are in displayed groups
    if (!walletRef.current) {
      return;
    }
    const taskIds = groups
      .flatMap((group) => group.tasks)
      .map((task) => task.id);
    if (!taskIds.includes(notification.step as OnboardingStep)) {
      return;
    }

    const rect = walletRef.current.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) {
      return;
    }

    const origin = {
      x: (rect.left + rect.width / 2) / window.innerWidth,
      y: (rect.top + rect.height / 2) / window.innerHeight,
    };
    const shared: ConfettiOptions = {
      particleCount: 50,
      spread: 70,
      shapes: ["square"],
      scalar: 1.2,
      startVelocity: 20,
      gravity: 0.6,
      decay: 0.92,
      ticks: 100,
      colors: AGPT_CONFETTI_COLORS,
      origin,
    };
    confetti({ ...shared, angle: 45 });
    confetti({ ...shared, angle: 135 });
  }

  // `handleNotification` is a fresh closure on every render. Route the listener
  // through a ref, refreshed after each render, so the subscription below stays
  // mounted for the lifetime of the hook instead of tearing down and
  // reconnecting each time.
  const notificationRef = useRef(handleNotification);

  useEffect(() => {
    notificationRef.current = handleNotification;
  });

  // WebSocket setup for onboarding notifications
  useEffect(() => {
    const detachMessage = api.onWebSocketMessage(
      "notification",
      (notification) => notificationRef.current(notification),
    );

    api.connectWebSocket();

    return () => {
      detachMessage();
    };
  }, [api]);

  // Wallet flash on credits change
  useEffect(() => {
    if (credits === prevCredits) {
      return;
    }
    setPrevCredits(credits);
    if (prevCredits === null) {
      return;
    }
    setFlash(true);
    setTimeout(() => {
      setFlash(false);
    }, 300);
  }, [credits, prevCredits]);

  return {
    state,
    groups,
    credits,
    formatCredits,
    flash,
    walletOpen,
    setWalletOpen,
    onWalletOpen,
    walletRef,
    completedCount,
    totalCount,
    topUpOpen,
    onAddCredits,
    onTopUpClose,
  };
}
