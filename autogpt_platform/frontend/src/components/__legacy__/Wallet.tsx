"use client";

import useCredits from "@/hooks/useCredits";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/__legacy__/ui/popover";
import { X } from "lucide-react";
import { Text } from "@/components/atoms/Text/Text";
import { PopoverClose } from "@radix-ui/react-popover";
import { TaskGroups } from "@/app/(no-navbar)/onboarding/components/WalletTaskGroups";
import { ScrollArea } from "./ui/scroll-area";
import { useOnboarding } from "@/providers/onboarding/onboarding-provider";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import * as party from "party-js";
import WalletRefill from "./WalletRefill";
import { OnboardingStep } from "@/lib/autogpt-server-api";
import { storage, Key as StorageKey } from "@/services/storage/local-storage";

export interface Task {
  id: OnboardingStep;
  name: string;
  amount: number;
  details: string;
  video?: string;
  progress?: {
    current: number;
    target: number;
  };
}

export interface TaskGroup {
  name: string;
  details: string;
  tasks: Task[];
}

export default function Wallet() {
  const { state, updateState } = useOnboarding();
  const groups = useMemo<TaskGroup[]>(() => {
    return [
      {
        name: "First Wins",
        details: "Kickstart your journey with quick wins.",
        tasks: [
          {
            id: "GET_RESULTS",
            name: "Complete onboarding and see your first agent's results",
            amount: 3,
            details: "",
          },
          {
            id: "MARKETPLACE_VISIT",
            name: "Go to Marketplace",
            amount: 1,
            details: "Click Marketplace in the top navigation",
            video: "/onboarding/marketplace-visit.mp4",
          },
          {
            id: "MARKETPLACE_ADD_AGENT",
            name: "Find and add an agent",
            amount: 1,
            details:
              "Search for an agent in the Marketplace and add it to your Library",
            video: "/onboarding/marketplace-add.mp4",
          },
          {
            id: "MARKETPLACE_RUN_AGENT",
            name: "Open the Library page and run an agent",
            amount: 1,
            details: "Go to the Library, open an agent you want, and run it",
            video: "/onboarding/marketplace-run.mp4",
          },
          {
            id: "BUILDER_SAVE_AGENT",
            name: "Place your first blocks and save your agent",
            amount: 1,
            details:
              "Open block library on the left and add a block to the canvas then save your agent",
            video: "/onboarding/builder-save.mp4",
          },
        ],
      },
      {
        name: "Consistency Challenge",
        details: "Build your rhythm and make agents part of your routine.",
        tasks: [
          {
            id: "RE_RUN_AGENT",
            name: "Re-run an agent",
            amount: 1,
            details: "Re-run an agent from the Library",
          },
          {
            id: "SCHEDULE_AGENT",
            name: "Schedule your first agent",
            amount: 1,
            details: "Schedule an agent to run on a recurring basis",
          },
          {
            id: "RUN_AGENTS",
            name: "Run 10 agents",
            amount: 3,
            details: "Run agents from Library or Builder 10 times",
            progress: {
              current: state?.agentRuns || 0,
              target: 10,
            },
          },
          {
            id: "RUN_3_DAYS",
            name: "Run agents 3 days in a row",
            amount: 1,
            details:
              "Run any agents from the Library or Builder for 3 days in a row",
            progress: {
              current: state?.consecutiveRunDays || 0,
              target: 3,
            },
          },
        ],
      },
      {
        name: "The Pro Playground",
        details: "Master powerful features to supercharge your workflow.",
        tasks: [
          {
            id: "TRIGGER_WEBHOOK",
            name: "Trigger an agent via webhook",
            amount: 1,
            details:
              "In the Builder, go to Settings and copy the Webhook URL. Use it to trigger your agent from another app.",
          },
          {
            id: "RUN_14_DAYS",
            name: "Run agents 14 days in a row",
            amount: 3,
            details:
              "Run any agents from the Library or Builder for 10 days in a row",
            progress: {
              current: state?.consecutiveRunDays || 0,
              target: 14,
            },
          },
          {
            id: "RUN_AGENTS_100",
            name: "Complete 100 agent runs",
            amount: 3,
            details: "Let your agents run and complete 100 tasks in total",
            progress: {
              current: state?.agentRuns || 0,
              target: 100,
            },
          },
        ],
      },
    ];
  }, [state]);

  const { credits, formatCredits, fetchCredits } = useCredits({
    fetchInitialCredits: true,
  });

  const [prevCredits, setPrevCredits] = useState<number | null>(credits);
  const [flash, setFlash] = useState(false);
  const [walletOpen, setWalletOpen] = useState(false);
  const [lastSeenCredits, setLastSeenCredits] = useState<number | null>(null);

  const totalCount = useMemo(() => {
    return groups.reduce((acc, group) => acc + group.tasks.length, 0);
  }, [groups]);

  // Get total completed count for all groups
  const [completedCount, setCompletedCount] = useState<number | null>(null);
  // Needed to show confetti when a new step is completed
  const [prevCompletedCount, setPrevCompletedCount] = useState<number | null>(
    null,
  );

  const walletRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!state) {
      return;
    }
    const completed = groups.reduce(
      (acc, group) =>
        acc +
        group.tasks.filter((task) => state?.completedSteps?.includes(task.id))
          .length,
      0,
    );
    setCompletedCount(completed);
  }, [groups, state?.completedSteps]);

  // Load last seen credits from localStorage once on mount
  useEffect(() => {
    const stored = storage.get(StorageKey.WALLET_LAST_SEEN_CREDITS);
    if (stored !== undefined && stored !== null) {
      const parsed = parseFloat(stored);
      if (!Number.isNaN(parsed)) setLastSeenCredits(parsed);
      else setLastSeenCredits(0);
    } else {
      setLastSeenCredits(0);
    }
  }, []);

  // Auto-open once if never shown, otherwise open only when credits increase beyond last seen
  useEffect(() => {
    if (typeof credits !== "number") return;
    // Open once for first-time users
    if (state && state.walletShown === false) {
      setWalletOpen(true);
      // Mark as shown so it won't reopen on every reload
      updateState({ walletShown: true });
      return;
    }
    // Open if user gained more credits than last acknowledged
    if (
      lastSeenCredits !== null &&
      credits > lastSeenCredits &&
      walletOpen === false
    ) {
      setWalletOpen(true);
    }
  }, [credits, lastSeenCredits, state?.walletShown, updateState, walletOpen]);

  const onWalletOpen = useCallback(async () => {
    if (!state?.walletShown) {
      updateState({ walletShown: true });
    }
    // Refresh credits when the wallet is opened
    fetchCredits();
  }, [state?.walletShown, updateState, fetchCredits]);

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
    // It's enough to check completed count,
    // because the order of completed steps is not important
    // If the count is the same, we don't need to do anything
    if (completedCount === null || completedCount === prevCompletedCount) {
      return;
    }
    // Otherwise, we need to set the new prevCompletedCount
    setPrevCompletedCount(completedCount);
    // If there was no previous count, we don't show confetti
    if (prevCompletedCount === null) {
      return;
    }
    // And emit confetti
    if (walletRef.current) {
      // Fix confetti appearing in the top left corner
      const rect = walletRef.current.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) {
        return;
      }
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
    completedCount,
    prevCompletedCount,
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
    <Popover
      open={walletOpen}
      onOpenChange={(open) => {
        setWalletOpen(open);
        if (!open) {
          // Persist the latest acknowledged credits so we only auto-open on future gains
          if (typeof credits === "number") {
            storage.set(StorageKey.WALLET_LAST_SEEN_CREDITS, String(credits));
            setLastSeenCredits(credits);
          }
        }
      }}
    >
      <PopoverTrigger asChild>
        <div className="relative inline-block">
          <button
            ref={walletRef}
            className={cn(
              "relative flex items-center gap-1 rounded-md bg-zinc-50 px-3 py-2 text-sm",
            )}
            onClick={onWalletOpen}
          >
            Earn credits{" "}
            <span className="text-sm font-semibold">
              {formatCredits(credits)}
            </span>
            {completedCount && completedCount < totalCount && (
              <span className="absolute right-1 top-1 h-2 w-2 rounded-full bg-violet-600"></span>
            )}
            <div className="absolute bottom-[-2.5rem] left-1/2 z-50 hidden -translate-x-1/2 transform whitespace-nowrap rounded-small bg-white px-4 py-2 shadow-md group-hover:block">
              <Text variant="body-medium">
                {completedCount} of {totalCount} rewards claimed
              </Text>
            </div>
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
          "rounded-xl border-zinc-100 bg-white shadow-[0_3px_3px] shadow-zinc-200",
        )}
      >
        {/* Header */}
        <div className="mx-1 flex items-center justify-between border-b border-zinc-200 pb-3">
          <span className="font-poppins font-medium text-zinc-900">
            Your credits
          </span>
          <div className="flex items-center text-sm text-violet-700">
            <div className="rounded-lg bg-violet-100 px-3 py-2">
              Earn credits{" "}
              <span className="font-semibold">{formatCredits(credits)}</span>
            </div>
            <PopoverClose aria-label="Close wallet">
              <X className="ml-2 h-5 w-5 text-zinc-800 hover:text-foreground" />
            </PopoverClose>
          </div>
        </div>
        <ScrollArea className="max-h-[85vh] overflow-y-auto">
          {/* Top ups */}
          {process.env.NEXT_PUBLIC_SHOW_BILLING_PAGE === "true" && (
            <WalletRefill />
          )}
          {/* Tasks */}
          <p className="mx-1 my-3 font-sans text-xs font-normal text-zinc-400">
            Complete the following tasks to earn more credits!
          </p>
          <TaskGroups groups={groups} />
        </ScrollArea>
      </PopoverContent>
    </Popover>
  );
}
