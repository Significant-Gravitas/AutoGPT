"use client";

import { useMountEffect } from "@/hooks/useMountEffect";
import { Key } from "@/services/storage/local-storage";

import { markShownToday, wasShownToday } from "./helpers";
import { useTopUpPrompt } from "./useTopUpPrompt";

export function DailyTopUpAutoOpener() {
  const { openTopUp } = useTopUpPrompt();

  useMountEffect(() => {
    if (!wasShownToday(Key.TOP_UP_MODAL_LAST_SHOWN)) {
      markShownToday(Key.TOP_UP_MODAL_LAST_SHOWN);
      openTopUp();
    }
  });

  return null;
}
