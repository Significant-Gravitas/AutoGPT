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
import { AUTOPILOT_MODE_OPTIONS, getModeOption } from "./helpers";
import { UnsupervisedConfirmDialog } from "./UnsupervisedConfirmDialog";
import { useAutopilotModeSelector } from "./useAutopilotModeSelector";

interface Props {
  sessionId: string | null;
  persistedMode: AutopilotMode | null;
}

export function AutopilotModeSelector({ sessionId, persistedMode }: Props) {
  const {
    mode,
    isDefault,
    selectMode,
    handleMenuClosed,
    isConfirmOpen,
    confirmUnsupervised,
    cancelUnsupervised,
  } = useAutopilotModeSelector({ sessionId, persistedMode });
  const current = getModeOption(mode);

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            type="button"
            aria-label={`Approval mode: ${current.label}. Change mode`}
            title={`Approval mode: ${current.label}`}
            className={cn(
              "inline-flex h-8 items-center justify-center gap-1 rounded-full text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400",
              isDefault
                ? "w-8 text-zinc-500 hover:bg-zinc-100 hover:text-zinc-700"
                : "px-2.5",
              mode === "ask_first" &&
                "bg-zinc-100 text-zinc-700 hover:bg-zinc-200",
              mode === "unsupervised" &&
                "bg-amber-50 text-amber-700 hover:bg-amber-100",
            )}
          >
            <Icon icon={current.icon} size={16} aria-hidden="true" />
            {!isDefault && <span>{current.label}</span>}
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          align="end"
          className="w-72"
          onCloseAutoFocus={handleMenuClosed}
        >
          <DropdownMenuLabel className="text-xs font-medium text-zinc-500">
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
                  <span className="font-medium text-zinc-900">
                    {option.label}
                  </span>
                  <span className="text-xs text-zinc-500">
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
      />
    </>
  );
}
