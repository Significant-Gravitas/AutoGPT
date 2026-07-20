"use client";

import { LocalPCWarning } from "@/app/(platform)/copilot/components/LocalPCWarning/LocalPCWarning";
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
  CaretDownIcon,
  CloudIcon,
  DesktopIcon,
  FolderOpenIcon,
} from "@phosphor-icons/react";
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
              className="inline-flex min-h-11 max-w-full items-center gap-2 rounded-full border border-zinc-200 bg-white px-3 py-2 text-sm font-medium text-zinc-800 shadow-sm outline-none hover:bg-zinc-50 focus-visible:ring-2 focus-visible:ring-violet-600 focus-visible:ring-offset-2"
            >
              {isLocal ? (
                <DesktopIcon
                  size={17}
                  weight="fill"
                  className="shrink-0 text-violet-600"
                  aria-hidden="true"
                />
              ) : (
                <CloudIcon
                  size={17}
                  weight="fill"
                  className="shrink-0 text-sky-600"
                  aria-hidden="true"
                />
              )}
              <span>{isLocal ? "Local PC" : "Cloud"}</span>
              <CaretDownIcon
                size={14}
                className="shrink-0 text-zinc-500"
                aria-hidden="true"
              />
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
                  <span className="text-xs text-zinc-500">
                    Run in an isolated hosted environment
                  </span>
                </div>
              </DropdownMenuRadioItem>
              <DropdownMenuRadioItem value="local" className="min-h-12">
                <div className="flex min-w-0 flex-col">
                  <span className="font-medium">Local PC</span>
                  <span className="text-xs text-zinc-500">
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
            className="inline-flex min-h-11 min-w-0 max-w-full items-center gap-2 rounded-full border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-700 shadow-sm outline-none hover:bg-zinc-50 focus-visible:ring-2 focus-visible:ring-violet-600 focus-visible:ring-offset-2"
          >
            <FolderOpenIcon
              size={17}
              weight="fill"
              className="shrink-0 text-amber-500"
              aria-hidden="true"
            />
            <span className="min-w-0 max-w-72 truncate">
              {picker.target.displayPath ?? "Choose a Folder"}
            </span>
          </button>
        ) : null}
      </div>

      {picker.error && picker.target.kind === "local" ? (
        <p
          role="alert"
          className="mb-2 px-3 text-left text-xs font-medium text-amber-800"
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
        onStale={picker.setError}
      />
    </>
  );
}
