"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { cn } from "@/lib/utils";
import type { AutopilotMode } from "../../../../autopilotModeStore";
import {
  AUTOPILOT_MODE_OPTIONS,
  getModeOption,
  unsupervisedChatDescription,
} from "./helpers";
import { UnsupervisedConfirmDialog } from "./UnsupervisedConfirmDialog";
import { useAutopilotModeSelector } from "./useAutopilotModeSelector";

interface Props {
  sessionId: string | null;
  persistedMode: AutopilotMode | null;
  disabled?: boolean;
}

export function AutopilotModeSelector({
  sessionId,
  persistedMode,
  disabled = false,
}: Props) {
  const {
    mode,
    isDefault,
    selectMode,
    isConfirmOpen,
    confirmUnsupervised,
    cancelUnsupervised,
    assistantName,
  } = useAutopilotModeSelector({ sessionId, persistedMode });
  const current = getModeOption(mode);

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild disabled={disabled}>
          <button
            type="button"
            aria-label={`Approval mode: ${current.label}. Change mode`}
            title={`Approval mode: ${current.label}`}
            className={cn(
              // TODO(tokens): no fitting role for focus-visible:ring-zinc-400 #ADADB3; #ADADB3 is foreground-subtle (a text colour)
              "inline-flex h-8 items-center justify-center gap-1 rounded-full text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400",
              isDefault
                ? "w-8 text-foreground-tertiary hover:bg-surface-hover hover:text-foreground-body"
                : "px-2.5",
              mode === "ask_first" &&
                "bg-surface-hover text-foreground-body hover:bg-surface-active",
              mode === "unsupervised" &&
                // TODO(tokens): no exact role for hover:bg-amber-100
                "bg-warning-surface text-warning-foreground hover:bg-amber-100",
              disabled && "opacity-40",
            )}
          >
            <Icon icon={current.icon} size={16} aria-hidden="true" />
            {!isDefault && <span>{current.label}</span>}
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-72">
          <DropdownMenuLabel className="text-xs font-medium text-foreground-tertiary">
            Approvals in this chat
          </DropdownMenuLabel>
          <DropdownMenuRadioGroup value={mode} onValueChange={selectMode}>
            {AUTOPILOT_MODE_OPTIONS.map((option) => (
              <DropdownMenuRadioItem
                key={option.value}
                value={option.value}
                className="items-start"
              >
                <span className="flex flex-col gap-0.5">
                  <span className="font-medium text-foreground-strong">
                    {option.label}
                  </span>
                  <span className="text-xs text-foreground-tertiary">
                    {option.description}
                  </span>
                </span>
              </DropdownMenuRadioItem>
            ))}
          </DropdownMenuRadioGroup>
        </DropdownMenuContent>
      </DropdownMenu>
      <UnsupervisedConfirmDialog
        isOpen={isConfirmOpen}
        onConfirm={confirmUnsupervised}
        onCancel={cancelUnsupervised}
        description={unsupervisedChatDescription(assistantName)}
      />
    </>
  );
}
