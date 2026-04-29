"use client";

import { useEffect, useState } from "react";

import {
  useGetV1GetAutoTopUp,
  usePostV1ConfigureAutoTopUp,
} from "@/app/api/__generated__/endpoints/credits/credits";
import type { AutoTopUpConfig } from "@/app/api/__generated__/models/autoTopUpConfig";

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
      setThreshold(config?.threshold ? (config.threshold / 100).toString() : "");
      setRefillAmount(config?.amount ? (config.amount / 100).toString() : "");
    }
  }, [open, config]);

  const thresholdValue = Number.parseFloat(threshold);
  const refillValue = Number.parseFloat(refillAmount);
  const isValid =
    Number.isFinite(thresholdValue) &&
    thresholdValue >= 5 &&
    Number.isFinite(refillValue) &&
    refillValue >= 5;

  async function save() {
    if (!isValid) return;
    await configureAutoTopUp({
      data: {
        amount: Math.round(refillValue * 100),
        threshold: Math.round(thresholdValue * 100),
      },
    });
    await refetch();
    setOpen(false);
  }

  async function disable() {
    await configureAutoTopUp({ data: { amount: 0, threshold: 0 } });
    await refetch();
    setOpen(false);
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
