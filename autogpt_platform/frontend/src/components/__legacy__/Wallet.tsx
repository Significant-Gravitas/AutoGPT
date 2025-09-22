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

export interface Task {
  id: OnboardingStep;
  name: string;
  amount: number;
  details: string;
  video?: string;
}

export interface TaskGroup {
  name: string;
  details: string;
  tasks: Task[];
  isOpen: boolean;
}

export default function Wallet() {
  const [groups, setGroups] = useState<TaskGroup[]>([
    {
      name: "First Wins",
      details: "Kickstart your journey with quick wins.",
      isOpen: true,
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
          details:
            "Go to the Library, open an agent you want, and run it",
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
      isOpen: true,
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
        },
        {
          id: "RUN_3_DAYS",
          name: "Run agents 3 days in a row",
          amount: 1,
          details: "Run any agents from the Library or Builder for 3 days in a row",
        },
      ],
    },
    {
      name: "The Pro Playground",
      details: "Master powerful features to supercharge your workflow.",
      isOpen: true,
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
          details: "Run any agents from the Library or Builder for 10 days in a row",
        },
        {
          id: "RUN_AGENTS_100",
          name: "Complete 100 agent runs",
          amount: 3,
          details: "Let your agents run and complete 100 tasks in total",
        },
      ],
    },
  ]);

  const { credits, formatCredits, fetchCredits } = useCredits({
    fetchInitialCredits: true,
  });

  const { state, updateState } = useOnboarding();
  const [prevCredits, setPrevCredits] = useState<number | null>(credits);
  const [flash, setFlash] = useState(false);
  const [walletOpen, setWalletOpen] = useState(state?.walletShown || false);

  const [stepsLength, setStepsLength] = useState<number | null>(
    state?.completedSteps?.length || null,
  );

  const totalCount = useMemo(() => {
    return groups.reduce((acc, group) => acc + group.tasks.length, 0);
  }, [groups]);

  // Get total completed count for all groups
  const completedCount = useMemo(() => {
    return groups.reduce(
      (acc, group) => acc + group.tasks.filter((task) => state?.completedSteps?.includes(task.id)).length,
      0,
    );
  }, [groups, state?.completedSteps]);

  const walletRef = useRef<HTMLButtonElement | null>(null);

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
    <Popover
      open={walletOpen}
      onOpenChange={setWalletOpen}
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
            {completedCount < totalCount && (
              <span className="absolute right-1 top-1 h-2 w-2 rounded-full bg-violet-600"></span>
            )}
            <div
              className="absolute bottom-[-2.5rem] left-1/2 z-50 hidden -translate-x-1/2 transform whitespace-nowrap rounded-small bg-white px-4 py-2 shadow-md group-hover:block"
            >
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
              Earn credits{" "}
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
          <TaskGroups
            groups={groups}
            setGroups={setGroups}
          />
        </ScrollArea>
      </PopoverContent>
    </Popover>
  );
}
