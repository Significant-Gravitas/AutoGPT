"use client";

import { ScrollArea } from "@/components/ui/scroll-area";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { PopoverClose } from "@radix-ui/react-popover";

import { TaskGroup } from "../helpers";
import { WalletRefill } from "./WalletRefill";
import { TaskGroups } from "./WalletTaskGroups";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  groups: TaskGroup[];
  formattedCredits: string;
}

export function WalletFullPanel({ groups, formattedCredits }: Props) {
  const isPaymentEnabled = useGetFlag(Flag.ENABLE_PLATFORM_PAYMENT);

  return (
    <>
      {/* Header */}
      <div className="mx-1 flex items-start justify-between gap-3 border-b border-zinc-200 pb-3">
        <div className="flex min-w-0 flex-col gap-1">
          <span className="font-poppins text-base font-semibold text-zinc-900">
            Automation Credits
          </span>
          <span className="font-sans text-xs text-zinc-500">
            Platform-only credits for automations. This is separate from your
            subscription and is not usable for plan fees.
          </span>
        </div>
        <div className="flex shrink-0 items-center text-sm text-violet-700">
          <div className="rounded-lg bg-violet-100 px-3 py-2">
            Earn credits{" "}
            <span className="font-semibold">{formattedCredits}</span>
          </div>
          <PopoverClose aria-label="Close wallet">
            <Icon
              icon={Cancel01Icon}
              className="ml-2 h-5 w-5 text-zinc-800 hover:text-foreground"
            />
          </PopoverClose>
        </div>
      </div>
      <ScrollArea className="max-h-[85vh] overflow-y-auto">
        {/* Top ups */}
        {isPaymentEnabled && <WalletRefill />}
        {/* Tasks */}
        <p className="mx-1 my-3 font-sans text-xs font-normal text-zinc-400">
          Complete the following tasks to earn more credits!
        </p>
        <TaskGroups groups={groups} />
      </ScrollArea>
    </>
  );
}
