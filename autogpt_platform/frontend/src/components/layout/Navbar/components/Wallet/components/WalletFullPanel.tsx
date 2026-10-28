"use client";

import { ScrollArea } from "@/components/molecules/ScrollArea/ScrollArea";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { PopoverClose } from "@radix-ui/react-popover";

import { TaskGroup } from "../helpers";
import { WalletRefill } from "./WalletRefill";
import { TaskGroups } from "./WalletTaskGroups";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  groups: TaskGroup[];
  formattedCredits: string;
}

export function WalletFullPanel({ groups, formattedCredits }: Props) {
  const isPaymentEnabled = useGetFlag(Flag.ENABLE_PLATFORM_PAYMENT);

  return (
    <>
      {/* Header */}
      <div className="mx-1 flex items-start justify-between gap-3 border-b border-border-base pb-3">
        <div className="flex min-w-0 flex-col gap-1">
          <Text variant="h5-semibold" as="span" tone="primary">
            Automation Credits
          </Text>
          <Text variant="small-tight" as="span" tone="muted">
            Platform-only credits for automations. This is separate from your
            subscription and is not usable for plan fees.
          </Text>
        </div>
        <div className="flex shrink-0 items-center text-body leading-5 text-accent-strong">
          <Text
            variant="body-tight"
            as="div"
            tone="inherit"
            // TODO(tokens): no exact role for bg-violet-100 (#EDE9FE)
            className="rounded-lg bg-violet-100 px-3 py-2"
          >
            Earn credits{" "}
            <Text as="span" variant="body-semibold-tight" tone="inherit">
              {formattedCredits}
            </Text>
          </Text>
          <PopoverClose aria-label="Close wallet">
            <Icon
              icon={Cancel01Icon}
              className="ml-2 h-5 w-5 text-foreground-emphasis hover:text-foreground"
            />
          </PopoverClose>
        </div>
      </div>
      <ScrollArea className="max-h-[85vh] overflow-y-auto">
        {/* Top ups */}
        {isPaymentEnabled && <WalletRefill />}
        {/* Tasks */}
        <Text variant="small-tight" as="p" tone="subtle" className="mx-1 my-3">
          Complete the following tasks to earn more credits!
        </Text>
        <TaskGroups groups={groups} />
      </ScrollArea>
    </>
  );
}
