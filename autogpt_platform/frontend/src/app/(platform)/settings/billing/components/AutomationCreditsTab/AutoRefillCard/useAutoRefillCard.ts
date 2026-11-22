"use client";

import { useEffect, useRef, useState } from "react";
import { v4 as uuidv4 } from "uuid";

import {
  useGetV1GetAutoTopUp,
  usePostV1ConfigureAutoTopUp,
} from "@/app/api/__generated__/endpoints/credits/credits";
import type { AutoTopUpConfig } from "@/app/api/__generated__/models/autoTopUpConfig";
import { toast } from "@/components/molecules/Toast/use-toast";

export function useAutoRefillCard() {
  const {
    data: config,
    isLoading,
    refetch,
  } = useGetV1GetAutoTopUp({
    query: {
      select: (res) => {
        const raw = res.data as AutoTopUpConfig | undefined;
        if (!raw) return undefined;
        if (!raw.amount && !raw.threshold) return undefined;
        return raw;
      },
    },
  });

  // Names this save, so a retry after a lost response can't charge again.
  // Kept through retries, replaced once it succeeds or the values change.
  const [saveKey, setSaveKey] = useState(newSaveKey);
  const { mutateAsync: configureAutoTopUp, isPending: isSaving } =
    usePostV1ConfigureAutoTopUp({
      request: { headers: { "Idempotency-Key": saveKey } },
    });

  const isEnabled = Boolean(config && (config.amount || config.threshold));

  const [open, setOpen] = useState(false);
  const [threshold, setThreshold] = useState("");
  const [refillAmount, setRefillAmount] = useState("");
  const initializedForOpenRef = useRef(false);
  // `isSaving` disables the buttons only after a re-render, so both clicks of
  // a double-click would get through, and a save below the threshold charges
  // the card.
  const configuringRef = useRef(false);

  useEffect(() => {
    if (!open) {
      initializedForOpenRef.current = false;
      return;
    }
    // Wait for the config fetch to settle before locking in defaults —
    // otherwise opening the dialog mid-load would seed "5/5" and skip
    // re-hydration when saved config arrives moments later.
    if (isLoading) return;
    // Initialize once per open transition so background refetches (e.g. on
    // window focus) don't clobber the user's in-progress edits.
    if (initializedForOpenRef.current) return;
    initializedForOpenRef.current = true;
    // Backend minimum is $5 — pre-fill with the floor so the form is valid
    // by default and matches the "$5 minimum" copy in the empty state.
    setThreshold(config?.threshold ? (config.threshold / 100).toString() : "5");
    setRefillAmount(config?.amount ? (config.amount / 100).toString() : "5");
  }, [open, config, isLoading]);

  const thresholdValue = Number.parseFloat(threshold);
  const refillValue = Number.parseFloat(refillAmount);
  // Use Number.isFinite (not Number.isInteger) so legacy non-whole-dollar
  // values (e.g. $7.50 stored as 750 cents) don't permanently disable save.
  // Math.round(value * 100) below converts back to integer cents.
  const isValid =
    Number.isFinite(thresholdValue) &&
    thresholdValue >= 5 &&
    Number.isFinite(refillValue) &&
    refillValue >= 5 &&
    // Backend rejects refill < threshold with 422 — gate it client-side too.
    refillValue >= thresholdValue;

  // Sends the request unless one is already in flight, and frees the guard as
  // soon as it settles, when `isSaving` re-enables the buttons.
  async function configureOnce(data: AutoTopUpConfig) {
    if (configuringRef.current) return false;
    configuringRef.current = true;
    try {
      await configureAutoTopUp({ data });
    } finally {
      configuringRef.current = false;
    }
    return true;
  }

  async function save() {
    if (!isValid) return;
    try {
      const sent = await configureOnce({
        amount: Math.round(refillValue * 100),
        threshold: Math.round(thresholdValue * 100),
      });
      if (!sent) return;
      setSaveKey(newSaveKey());
      await refetch();
      setOpen(false);
    } catch (error) {
      toast({
        title: "Couldn't save auto top-up",
        description:
          error instanceof Error
            ? error.message
            : "Auto top-up settings weren't saved. Please try again.",
        variant: "destructive",
      });
    }
  }

  async function disable() {
    try {
      const sent = await configureOnce({ amount: 0, threshold: 0 });
      if (!sent) return;
      await refetch();
      setOpen(false);
    } catch (error) {
      toast({
        title: "Couldn't disable auto top-up",
        description:
          error instanceof Error
            ? error.message
            : "Auto top-up couldn't be disabled. Please try again.",
        variant: "destructive",
      });
    }
  }

  // Different values are a different save.
  function changeThreshold(value: string) {
    setThreshold(value);
    setSaveKey(newSaveKey());
  }

  function changeRefillAmount(value: string) {
    setRefillAmount(value);
    setSaveKey(newSaveKey());
  }

  return {
    config,
    isEnabled,
    isLoading,
    isSaving,
    open,
    setOpen,
    threshold,
    setThreshold: changeThreshold,
    refillAmount,
    setRefillAmount: changeRefillAmount,
    isValid,
    save: () => void save(),
    disable: () => void disable(),
  };
}

function newSaveKey() {
  // uuidv4({}) rather than crypto.randomUUID, which Chrome added only in 92.
  return uuidv4({});
}
