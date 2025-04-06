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
import { useCallback, useEffect, useRef } from "react";
import { cn } from "@/lib/utils";
import * as party from "party-js";

export default function Wallet() {
  const { credits, formatCredits, fetchCredits } = useCredits({
    fetchInitialCredits: true,
  });
  const { state, updateState } = useOnboarding();
  const walletRef = useRef<HTMLButtonElement | null>(null);

  const onWalletOpen = useCallback(async () => {
    if (state?.notificationDot) {
      updateState({ notificationDot: false });
    }
    // Refresh credits when the wallet is opened
    fetchCredits();
  }, [state?.notificationDot, updateState, fetchCredits]);

  const fadeOut = new party.ModuleBuilder()
    .drive("opacity")
    .by((t) => 1 - t)
    .through("lifetime")
    .build();

  useEffect(() => {
    // Check if there are any completed tasks (state?.completedTasks) that
    // are not in the state?.notified array and play confetti if so
    const pending = state?.completedSteps
      .filter((step) => !state?.notified.includes(step))
      // Ignore steps that are not relevant for notifications
      .filter(
        (step) =>
          step !== "WELCOME" &&
          step !== "USAGE_REASON" &&
          step !== "INTEGRATIONS" &&
          step !== "AGENT_CHOICE" &&
          step !== "AGENT_NEW_RUN" &&
          step !== "AGENT_INPUT",
      );
    if ((pending?.length || 0) > 0 && walletRef.current) {
      party.confetti(walletRef.current, {
        count: 30,
        spread: 120,
        shapes: ["square", "circle"],
        size: party.variation.range(1, 2),
        speed: party.variation.range(200, 300),
        modules: [fadeOut],
      });
    }
  }, [state?.completedSteps, state?.notified]);

  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          ref={walletRef}
          className="relative flex items-center gap-1 rounded-md bg-zinc-200 px-3 py-2 text-sm transition-colors duration-200 hover:bg-zinc-300"
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
      </PopoverTrigger>
      <PopoverContent
        className={cn(
          "absolute -right-[7.9rem] -top-[3.2rem] z-50 w-[28.5rem] px-[0.625rem] py-2",
          "rounded-xl border-zinc-200 bg-zinc-50 shadow-[0_3px_3px] shadow-zinc-300",
        )}
      >
        <div>
          <div className="mx-1 flex items-center justify-between border-b border-zinc-300 pb-2">
            <span className="font-poppins font-medium text-zinc-900">
              Your wallet
            </span>
            <div className="flex items-center font-inter text-sm font-semibold text-violet-700">
              <div className="rounded-lg bg-violet-100 px-3 py-2">
                Wallet{" "}
                <span className="font-semibold">{formatCredits(credits)}</span>
              </div>
              <PopoverClose>
                <X className="ml-[2.8rem] h-5 w-5 text-zinc-800 hover:text-foreground" />
              </PopoverClose>
            </div>
          </div>
          <p className="mx-1 mt-3 font-inter text-xs text-muted-foreground text-zinc-400">
            Complete the following tasks to earn more credits!
          </p>
        </div>
        <ScrollArea className="max-h-[80vh] overflow-y-auto">
          <TaskGroups />
        </ScrollArea>
      </PopoverContent>
    </Popover>
  );
}
