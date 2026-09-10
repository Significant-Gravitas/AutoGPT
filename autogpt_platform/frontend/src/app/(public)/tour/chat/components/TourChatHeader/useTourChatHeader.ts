"use client";

import { useToast } from "@/components/molecules/Toast/use-toast";
import { useRef, useState } from "react";
import { trackTourCtaClick } from "../../tracking";
import { buildTourShareUrl, copyTextToClipboard } from "./helpers";

const COPIED_RESET_MS = 2000;

export function useTourChatHeader() {
  const [isCopied, setIsCopied] = useState(false);
  const resetTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const { toast } = useToast();

  async function handleShare() {
    trackTourCtaClick("share");
    const copied = await copyTextToClipboard(buildTourShareUrl());
    if (!copied) {
      toast({
        title: "Couldn't copy the link",
        description: buildTourShareUrl(),
        variant: "destructive",
      });
      return;
    }
    setIsCopied(true);
    if (resetTimer.current) clearTimeout(resetTimer.current);
    resetTimer.current = setTimeout(() => setIsCopied(false), COPIED_RESET_MS);
  }

  return { isCopied, handleShare };
}
