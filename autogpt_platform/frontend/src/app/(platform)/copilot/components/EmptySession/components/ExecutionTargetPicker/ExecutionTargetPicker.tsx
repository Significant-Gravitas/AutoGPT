"use client";

import { LocalPCWarning } from "@/app/(platform)/copilot/components/LocalPCWarning/LocalPCWarning";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import {
  ArrowDown01Icon,
  CloudIcon,
  ComputerIcon,
  FolderOpenIcon,
} from "@hugeicons/core-free-icons";
import { useState } from "react";
import { LocalFolderPicker } from "./components/LocalFolderPicker/LocalFolderPicker";
import { useExecutionTargetPicker } from "./useExecutionTargetPicker";

export function ExecutionTargetPicker() {
  const picker = useExecutionTargetPicker();
  const [warningResolved, setWarningResolved] = useState(false);
  const isLocal = picker.target.kind === "local";

  function handleTargetChange(value: string) {
    if (value === "local") picker.selectLocal();
    else picker.selectCloud();
  }

  return (
    <>
      <div className="mb-2 flex min-w-0 flex-wrap items-center gap-2 px-2 text-left">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              type="button"
              aria-label={`Execution target: ${isLocal ? "Local PC" : "Cloud"}`}
              className="inline-flex min-h-11 max-w-full items-center gap-2 rounded-full border border-border-base bg-surface px-3 py-2 text-sm font-medium text-foreground-emphasis shadow-sm outline-none hover:bg-surface-muted focus-visible:ring-2 focus-visible:ring-brand focus-visible:ring-offset-2"
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
            <DropdownMenuLabel>Run This Chat In</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuRadioGroup
              value={picker.target.kind}
              onValueChange={handleTargetChange}
            >
              <DropdownMenuRadioItem value="cloud" className="min-h-12">
                <div className="flex min-w-0 flex-col">
                  <span className="font-medium">Cloud</span>
                  <span className="text-xs text-foreground-tertiary">
                    Run in an isolated hosted environment
                  </span>
                </div>
              </DropdownMenuRadioItem>
              <DropdownMenuRadioItem value="local" className="min-h-12">
                <div className="flex min-w-0 flex-col">
                  <span className="font-medium">Local PC</span>
                  <span className="text-xs text-foreground-tertiary">
                    Run on a connected computer and folder
                  </span>
                </div>
              </DropdownMenuRadioItem>
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
            className="inline-flex min-h-11 max-w-full min-w-0 items-center gap-2 rounded-full border border-border-base bg-surface px-3 py-2 text-sm text-foreground-body shadow-sm outline-none hover:bg-surface-muted focus-visible:ring-2 focus-visible:ring-brand focus-visible:ring-offset-2"
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
          className="mb-2 px-3 text-left text-xs font-medium text-warning-foreground"
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
        isOpen={isLocal && warningResolved && picker.isOpen}
        machines={picker.machines}
        selectedMachineID={
          picker.target.kind === "local" ? picker.target.machineID : null
        }
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
