"use client";

import { LocalPCWarning } from "@/app/(platform)/copilot/components/LocalPCWarning/LocalPCWarning";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import {
  ArrowDown01Icon,
  CloudIcon,
  ComputerIcon,
  FolderOpenIcon,
} from "@hugeicons/core-free-icons";
import { cn } from "@/lib/utils";
import { useState } from "react";
import { LocalFolderPicker } from "./components/LocalFolderPicker/LocalFolderPicker";
import { useAnotherComputer } from "./useAnotherComputer";
import { useExecutionTargetPicker } from "./useExecutionTargetPicker";

interface Props {
  /** Inside the composer tray: a quiet text button, like the tray's footer. */
  inTray?: boolean;
}

const TARGET_OPTIONS = [
  {
    value: "cloud",
    label: "Cloud",
    description: "Run in an isolated hosted environment",
    icon: CloudIcon,
    tone: "text-info",
  },
  {
    value: "local",
    label: "Local PC",
    description: "Run on a connected computer and folder",
    icon: ComputerIcon,
    tone: "text-brand",
  },
];

const PILL_BUTTON =
  "min-h-11 gap-2 rounded-full border border-border-base bg-surface px-3 py-2 text-body shadow-sm hover:bg-surface-muted focus-visible:ring-offset-2";

const TRAY_BUTTON =
  "h-7 gap-1.5 rounded-lg px-1.5 text-body-sm font-normal hover:bg-surface-tray";

export function ExecutionTargetPicker({ inTray = false }: Props) {
  const picker = useExecutionTargetPicker();
  const [warningResolved, setWarningResolved] = useState(false);
  const isLocal = picker.target.kind === "local";
  const isFolderDialogOpen = isLocal && warningResolved && picker.isOpen;
  const selectedMachineID =
    picker.target.kind === "local" ? picker.target.machineID : null;
  const anotherComputer = useAnotherComputer({
    isOpen: isFolderDialogOpen,
    machines: picker.machines,
    selectedMachineID,
  });

  function handleTargetChange(value: string) {
    if (value === "local") picker.selectLocal();
    else picker.selectCloud();
  }

  return (
    <>
      <div
        className={cn(
          "flex min-w-0 flex-wrap items-center text-left",
          inTray ? "min-h-9 gap-1 pr-2 pl-1.5" : "mb-2 gap-2 px-2",
        )}
      >
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              type="button"
              aria-label={`Execution target: ${isLocal ? "Local PC" : "Cloud"}`}
              className={cn(
                "inline-flex max-w-full items-center outline-none focus-visible:ring-2 focus-visible:ring-brand",
                inTray
                  ? cn(TRAY_BUTTON, "text-foreground-text")
                  : cn(PILL_BUTTON, "font-medium text-foreground-emphasis"),
              )}
            >
              {isLocal ? (
                <span className="flex shrink-0 text-brand">
                  <Icon icon={ComputerIcon} size={17} aria-hidden="true" />
                </span>
              ) : (
                <span className="flex shrink-0 text-info">
                  <Icon icon={CloudIcon} size={17} aria-hidden="true" />
                </span>
              )}
              <span>{isLocal ? "Local PC" : "Cloud"}</span>
              <span className="flex shrink-0 text-foreground-tertiary">
                <Icon icon={ArrowDown01Icon} size={14} aria-hidden="true" />
              </span>
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="w-72">
            <DropdownMenuLabel>Run this chat in</DropdownMenuLabel>
            <DropdownMenuRadioGroup
              value={picker.target.kind}
              onValueChange={handleTargetChange}
            >
              {TARGET_OPTIONS.map((option) => (
                <DropdownMenuRadioItem
                  key={option.value}
                  value={option.value}
                  className="items-start"
                >
                  <span className="flex min-w-0 flex-1 items-start gap-2.5">
                    <span
                      className={cn(
                        "flex size-8 shrink-0 items-center justify-center rounded-lg bg-surface-muted",
                        option.tone,
                      )}
                    >
                      <Icon icon={option.icon} size={17} aria-hidden="true" />
                    </span>
                    <span className="flex min-w-0 flex-col gap-0.5">
                      <span className="font-medium text-foreground-strong">
                        {option.label}
                      </span>
                      <span className="text-caption text-foreground-tertiary">
                        {option.description}
                      </span>
                    </span>
                  </span>
                </DropdownMenuRadioItem>
              ))}
            </DropdownMenuRadioGroup>
          </DropdownMenuContent>
        </DropdownMenu>

        {picker.target.kind === "local" ? (
          <button
            type="button"
            onClick={() => picker.setOpen(true)}
            aria-label={
              picker.target.displayPath
                ? `Local folder: ${picker.target.displayPath}. Change folder`
                : "Choose a Local PC folder"
            }
            className={cn(
              "inline-flex max-w-full min-w-0 items-center outline-none focus-visible:ring-2 focus-visible:ring-brand",
              inTray
                ? cn(TRAY_BUTTON, "text-foreground-text")
                : cn(PILL_BUTTON, "text-foreground-body"),
            )}
          >
            <span className="flex shrink-0 text-warning">
              <Icon icon={FolderOpenIcon} size={17} aria-hidden="true" />
            </span>
            <span className="max-w-72 min-w-0 truncate">
              {picker.target.displayPath ?? "Choose a Folder"}
            </span>
          </button>
        ) : null}
      </div>

      {picker.error && picker.target.kind === "local" ? (
        <p
          role="alert"
          className="mb-2 px-3 text-left text-caption font-medium text-warning-foreground"
        >
          {picker.error}
        </p>
      ) : null}

      {isLocal ? (
        <LocalPCWarning
          onResolved={setWarningResolved}
          onCancel={picker.selectCloud}
        />
      ) : null}

      <LocalFolderPicker
        key={
          picker.target.kind === "local"
            ? `${picker.target.machineID ?? "none"}:${picker.target.connectionID ?? "none"}`
            : "cloud"
        }
        isOpen={isFolderDialogOpen}
        machines={picker.machines}
        selectedMachineID={selectedMachineID}
        anotherComputer={anotherComputer}
        isLoadingMachines={picker.isLoadingMachines}
        isMachinesError={picker.isMachinesError}
        isRefreshingMachines={picker.isRefreshingMachines}
        onOpenChange={picker.setOpen}
        onSelectMachine={picker.selectMachine}
        onRefreshMachines={() => void picker.retryMachines()}
        onSelectDirectory={(machine, browseID, directory) =>
          picker.selectDirectory(
            machine,
            browseID,
            directory.directory_ref,
            directory.path,
          )
        }
        onStale={picker.invalidateDirectorySelection}
      />
    </>
  );
}
