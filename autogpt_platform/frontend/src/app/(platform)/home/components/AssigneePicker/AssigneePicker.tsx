"use client";

import { RecipientChip } from "@/app/(platform)/copilot/components/ChatInput/components/RecipientChip";
import { useExpertMap } from "@/app/(platform)/copilot/useExpertMap";
import { assigneeOptions, OTTO_OPTION } from "./helpers";

interface Props {
  expertId: string | null;
  onSelect: (expertId: string | null) => void;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  onCloseAutoFocus?: (event: Event) => void;
  compact?: boolean;
}

export function AssigneePicker({
  expertId,
  onSelect,
  open,
  onOpenChange,
  onCloseAutoFocus,
  compact = false,
}: Props) {
  const { isExpertsEnabled, activeExperts } = useExpertMap();

  if (!isExpertsEnabled) return null;

  const options = assigneeOptions(activeExperts);
  const recipient =
    options.find((option) => option.id === expertId) ?? OTTO_OPTION;

  return (
    <RecipientChip
      recipient={recipient}
      options={options}
      onSelect={onSelect}
      open={open}
      onOpenChange={onOpenChange}
      onCloseAutoFocus={onCloseAutoFocus}
      ariaLabel={`Assignee: ${recipient.name} — change assignee`}
      hideName={compact}
      className={
        compact
          ? "ml-0 h-8 gap-0.5 rounded-full border-transparent pl-1 pr-1.5 shadow-none hover:border-zinc-200"
          : "ml-0 h-7 rounded-full pl-0.5 text-xs shadow-none"
      }
    />
  );
}
