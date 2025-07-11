"use client";

import useCredits from "@/hooks/useCredits";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { X } from "lucide-react";
import { PopoverClose } from "@radix-ui/react-popover";
import { TaskGroups } from "../onboarding/WalletTaskGroups";
import { ScrollArea } from "../ui/scroll-area";
import { useOnboarding } from "../onboarding/onboarding-provider";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import * as party from "party-js";
import WalletRefill from "./WalletRefill";

export default function Wallet() {
  const { credits, formatCredits, fetchCredits } = useCredits({
    fetchInitialCredits: true,
  });
  const { state, updateState } = useOnboarding();
  const [prevCredits, setPrevCredits] = useState<number | null>(credits);
  const [flash, setFlash] = useState(false);
  const [stepsLength, setStepsLength] = useState<number | null>(
    state?.completedSteps?.length || null,
  );
  const walletRef = useRef<HTMLButtonElement | null>(null);

  const onWalletOpen = useCallback(async () => {
    if (state?.notificationDot) {
      updateState({ notificationDot: false });
    }
    // Refresh credits when the wallet is opened
    fetchCredits();
  }, [state?.notificationDot, updateState, fetchCredits]);

  const fadeOut = useMemo(
    () =>
      new party.ModuleBuilder()
        .drive("opacity")
        .by((t) => 1 - t)
        .through("lifetime")
        .build(),
    [],
  );

  // Confetti effect on the wallet button
  useEffect(() => {
    if (!state?.completedSteps) {
      return;
    }
    // If we haven't set the length yet, just set it and return
    if (stepsLength === null) {
      setStepsLength(state?.completedSteps?.length);
      return;
    }
    // It's enough to compare array lengths,
    // because the order of completed steps is not important
    // If the length is the same, we don't need to do anything
    if (state?.completedSteps?.length === stepsLength) {
      return;
    }
    // Otherwise, we need to set the new length
    setStepsLength(state?.completedSteps?.length);
    // And make confetti
    if (walletRef.current) {
      setTimeout(() => {
        fetchCredits();
        party.confetti(walletRef.current!, {
          count: 30,
          spread: 120,
          shapes: ["square", "circle"],
          size: party.variation.range(1, 2),
          speed: party.variation.range(200, 300),
          modules: [fadeOut],
        });
      }, 800);
    }
  }, [
    state?.completedSteps,
    state?.notified,
    fadeOut,
    fetchCredits,
    stepsLength,
    walletRef,
  ]);

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

  return (
    <Popover>
      <PopoverTrigger asChild>
        <div className="relative inline-block">
          <button
            ref={walletRef}
            className={cn(
              "relative flex items-center gap-1 rounded-md bg-zinc-50 px-3 py-2 text-sm",
            )}
            onClick={onWalletOpen}
          >
            Wallet{" "}
            <span className="text-sm font-semibold">
              {formatCredits(credits)}
            </span>
            {state?.notificationDot && (
              <span className="absolute right-1 top-1 h-2 w-2 rounded-full bg-violet-600"></span>
            )}
          </button>
          <div
            className={cn(
              "pointer-events-none absolute inset-0 rounded-md bg-violet-400 duration-2000 ease-in-out",
              flash ? "opacity-50 duration-0" : "opacity-0",
            )}
          />
        </div>
      </PopoverTrigger>
      <PopoverContent
        className={cn(
          "absolute -right-[7.9rem] -top-[3.2rem] z-50 w-[28.5rem] px-[0.625rem] py-2",
          "rounded-xl border-zinc-200 bg-zinc-50 shadow-[0_3px_3px] shadow-zinc-300",
        )}
      >
        {/* Header */}
        <div className="mx-1 flex items-center justify-between border-b border-zinc-300 pb-2">
          <span className="font-poppins font-medium text-zinc-900">
            Your wallet
          </span>
          <div className="flex items-center text-sm font-semibold text-violet-700">
            <div className="rounded-lg bg-violet-100 px-3 py-2">
              Wallet{" "}
              <span className="font-semibold">{formatCredits(credits)}</span>
            </div>
            <PopoverClose>
              <X className="ml-[2.8rem] h-5 w-5 text-zinc-800 hover:text-foreground" />
            </PopoverClose>
          </div>
        </div>
        <ScrollArea className="max-h-[85vh] overflow-y-auto">
          {/* Top ups */}
          {process.env.NEXT_PUBLIC_SHOW_BILLING_PAGE === "true" && (
            <WalletRefill />
          )}
          {/* Tasks */}
          <p className="mx-1 mt-4 font-sans text-xs font-medium text-violet-700">
            Onboarding tasks
          </p>
          <p className="mx-1 my-1 font-sans text-xs font-normal text-zinc-500">
            Complete the following tasks to earn more credits!
          </p>
          <TaskGroups />
        </ScrollArea>
      </PopoverContent>
    </Popover>
  );
}
