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
import { useCallback } from "react";

export default function Wallet() {
  const { credits, formatCredits, fetchCredits } = useCredits({
    fetchInitialCredits: true,
  });
  const { state, updateState } = useOnboarding();

  const onWalletOpen = useCallback(async () => {
    if (state?.notificationDot) {
      updateState({ notificationDot: false });
    }
    // Refresh credits when the wallet is opened
    fetchCredits();
  }, [state?.notificationDot, updateState, fetchCredits]);

  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
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
      <PopoverContent className="absolute -right-[8.8rem] -top-[3.1rem] z-50 w-[28.5rem] rounded-xl border-[0.05rem] border-b-[0.2rem] border-zinc-200 bg-zinc-50 p-3 shadow-none shadow-zinc-400">
        <div>
          <div className="mb-4 flex items-center justify-between">
            <span className="font-poppins font-medium text-zinc-900">
              Your wallet
            </span>
            <div className="flex items-center gap-1 font-inter text-sm font-semibold text-violet-700">
              Wallet
              <span className="font-semibold">{formatCredits(credits)}</span>
              <PopoverClose>
                <X className="ml-[3.4rem] h-5 w-5 text-zinc-800 hover:text-foreground" />
              </PopoverClose>
            </div>
          </div>
          <p className="mt-6 font-inter text-xs text-muted-foreground text-zinc-400">
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
