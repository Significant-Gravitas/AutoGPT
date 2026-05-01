"use client";

import { useEffect, useState } from "react";

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

  const { mutateAsync: configureAutoTopUp, isPending: isSaving } =
    usePostV1ConfigureAutoTopUp();

  const isEnabled = Boolean(config && (config.amount || config.threshold));

  const [open, setOpen] = useState(false);
  const [threshold, setThreshold] = useState("");
  const [refillAmount, setRefillAmount] = useState("");

  useEffect(() => {
    if (open) {
      setThreshold(
        config?.threshold ? (config.threshold / 100).toString() : "",
      );
      setRefillAmount(config?.amount ? (config.amount / 100).toString() : "");
    }
  }, [open, config]);

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

  async function save() {
    if (!isValid) return;
    try {
      await configureAutoTopUp({
        data: {
          amount: Math.round(refillValue * 100),
          threshold: Math.round(thresholdValue * 100),
        },
      });
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
      await configureAutoTopUp({ data: { amount: 0, threshold: 0 } });
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

  return {
    config,
    isEnabled,
    isLoading,
    isSaving,
    open,
    setOpen,
    threshold,
    setThreshold,
    refillAmount,
    setRefillAmount,
    isValid,
    save: () => void save(),
    disable: () => void disable(),
  };
}
