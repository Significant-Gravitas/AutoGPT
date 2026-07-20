"use client";

import type { DirectoryEntry } from "@/app/api/__generated__/models/directoryEntry";
import type { ExecutorMachine } from "@/app/api/__generated__/models/executorMachine";
import { Button } from "@/components/atoms/Button/Button";
import { Select } from "@/components/atoms/Select/Select";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  CaretLeftIcon,
  CaretRightIcon,
  FolderIcon,
  HouseIcon,
  WarningCircleIcon,
} from "@phosphor-icons/react";
import { useEffect, useRef, useState } from "react";
import { LocalExecutorSetup } from "../LocalExecutorSetup/LocalExecutorSetup";
import { useLocalFolderPicker } from "./useLocalFolderPicker";

interface Props {
  isOpen: boolean;
  machines: ExecutorMachine[];
  selectedMachineID: string | null;
  isLoadingMachines: boolean;
  isMachinesError: boolean;
  isRefreshingMachines: boolean;
  onOpenChange: (open: boolean) => void;
  onSelectMachine: (machineID: string) => void;
  onRefreshMachines: () => void;
  onSelectDirectory: (
    machine: ExecutorMachine,
    browseID: string,
    directory: DirectoryEntry,
  ) => void;
  onStale: (message: string) => void;
}

export function LocalFolderPicker({
  isOpen,
  machines,
  selectedMachineID,
  isLoadingMachines,
  isMachinesError,
  isRefreshingMachines,
  onOpenChange,
  onSelectMachine,
  onRefreshMachines,
  onSelectDirectory,
  onStale,
}: Props) {
  const machine =
    machines.find((item) => item.machine_id === selectedMachineID) ?? null;
  const browser = useLocalFolderPicker({ isOpen, machine, onStale });
  const previousMachineCountRef = useRef(machines.length);
  const [machineAnnouncement, setMachineAnnouncement] = useState("");

  useEffect(
    function announceNewConnection() {
      const previousCount = previousMachineCountRef.current;
      previousMachineCountRef.current = machines.length;
      if (!isOpen || previousCount !== 0 || machines.length === 0) return;

      setMachineAnnouncement(
        `${machines[0].display_name} connected. Choose a computer and folder.`,
      );
      const frame = requestAnimationFrame(() => {
        document.getElementById("local-executor-machine")?.focus();
      });
      return () => cancelAnimationFrame(frame);
    },
    [isOpen, machines],
  );

  function handleUseFolder() {
    if (!machine || !browser.directory?.current) return;
    onSelectDirectory(
      machine,
      browser.directory.browse_id,
      browser.directory.current,
    );
  }

  return (
    <Dialog
      title="Choose a Folder on Your Local PC"
      styling={{ maxWidth: "44rem", minWidth: "auto" }}
      controlled={{ isOpen, set: onOpenChange }}
    >
      <Dialog.Content>
        <div className="flex min-h-[24rem] flex-col gap-4">
          <span role="status" aria-live="polite" className="sr-only">
            {machineAnnouncement}
          </span>
          {isLoadingMachines ? (
            <div role="status" aria-live="polite" className="space-y-3">
              <span className="sr-only">Looking for connected computers…</span>
              <Skeleton className="h-11 w-full rounded-xl" />
              <Skeleton className="h-16 w-full rounded-xl" />
              <Skeleton className="h-16 w-full rounded-xl" />
            </div>
          ) : isMachinesError ? (
            <div
              role="alert"
              className="flex flex-col items-center gap-3 rounded-2xl border border-red-200 bg-red-50 p-6 text-center"
            >
              <WarningCircleIcon
                size={24}
                weight="fill"
                className="text-red-700"
                aria-hidden="true"
              />
              <Text variant="body-medium" className="text-red-900">
                Could Not Check Your Computers
              </Text>
              <Text variant="small" className="text-red-800">
                Check your connection and try again.
              </Text>
              <Button
                type="button"
                variant="secondary"
                size="small"
                onClick={onRefreshMachines}
              >
                Retry
              </Button>
            </div>
          ) : machines.length === 0 ? (
            <LocalExecutorSetup
              isRefreshing={isRefreshingMachines}
              onRefresh={onRefreshMachines}
            />
          ) : (
            <>
              <Select
                id="local-executor-machine"
                label="Computer"
                value={machine?.machine_id ?? ""}
                onValueChange={onSelectMachine}
                options={machines.map((item) => ({
                  value: item.machine_id,
                  label: `${item.display_name} · ${platformLabel(item.platform)}`,
                }))}
                wrapperClassName="!mb-0"
              />

              <FolderPath
                history={browser.history}
                currentPath={browser.directory?.current?.path ?? null}
                canGoBack={
                  !!browser.directory &&
                  (!!browser.directory.current ||
                    browser.directory.parent_ref !== null)
                }
                onBack={browser.openParent}
                onOpenBreadcrumb={browser.openBreadcrumb}
              />

              <div
                aria-busy={browser.isLoading}
                className="min-h-0 flex-1 rounded-2xl border border-zinc-200 bg-white"
              >
                {browser.isLoading ? (
                  <div
                    role="status"
                    aria-live="polite"
                    className="flex flex-col gap-2 p-3"
                  >
                    <span className="sr-only">Loading folders…</span>
                    {Array.from({ length: 4 }, (_, index) => (
                      <Skeleton
                        key={index}
                        className="h-12 w-full rounded-xl"
                      />
                    ))}
                  </div>
                ) : browser.error ? (
                  <div
                    role="alert"
                    className="flex min-h-48 flex-col items-center justify-center gap-3 p-6 text-center"
                  >
                    <WarningCircleIcon
                      size={24}
                      weight="fill"
                      className="text-amber-700"
                      aria-hidden="true"
                    />
                    <Text variant="small-medium" className="text-zinc-900">
                      {browser.error}
                    </Text>
                    <Button
                      type="button"
                      variant="secondary"
                      size="small"
                      onClick={browser.retry}
                    >
                      Retry
                    </Button>
                  </div>
                ) : (browser.directory?.entries.length ?? 0) === 0 ? (
                  <div
                    role="status"
                    className="flex min-h-48 items-center justify-center p-6 text-center text-sm text-zinc-500"
                  >
                    This folder has no subfolders.
                  </div>
                ) : (
                  <ul
                    aria-label="Folders"
                    className="max-h-[22rem] overflow-y-auto p-2"
                  >
                    {browser.directory?.entries.map((entry) => (
                      <li key={entry.directory_ref}>
                        <button
                          type="button"
                          onClick={() => browser.openDirectory(entry)}
                          className="flex min-h-11 w-full min-w-0 items-center gap-3 rounded-xl px-3 py-2 text-left text-sm text-zinc-800 outline-none hover:bg-zinc-100 focus-visible:ring-2 focus-visible:ring-violet-600 focus-visible:ring-offset-1"
                        >
                          <FolderIcon
                            size={20}
                            weight="fill"
                            className="shrink-0 text-violet-500"
                            aria-hidden="true"
                          />
                          <span className="min-w-0 flex-1 truncate">
                            {entry.name}
                          </span>
                          <CaretRightIcon
                            size={16}
                            className="shrink-0 text-zinc-400"
                            aria-hidden="true"
                          />
                        </button>
                      </li>
                    ))}
                    {browser.directory?.next_cursor ? (
                      <li className="flex justify-center p-2">
                        <Button
                          type="button"
                          variant="secondary"
                          size="small"
                          loading={browser.isLoadingMore}
                          onClick={() => void browser.loadMore()}
                        >
                          Load More Folders
                        </Button>
                      </li>
                    ) : null}
                  </ul>
                )}
              </div>

              {browser.directory?.truncated &&
              !browser.directory.next_cursor ? (
                <Text role="status" variant="small" className="text-amber-800">
                  This folder is too large to show completely. Choose a visible
                  folder or narrow it on your computer.
                </Text>
              ) : null}
            </>
          )}
        </div>

        <Dialog.Footer className="flex-wrap justify-end">
          <Button
            type="button"
            variant="secondary"
            size="small"
            onClick={() => onOpenChange(false)}
          >
            Cancel
          </Button>
          {machines.length > 0 ? (
            <Button
              type="button"
              variant="primary"
              size="small"
              disabled={!browser.directory?.current || browser.isLoading}
              onClick={handleUseFolder}
            >
              Use This Folder
            </Button>
          ) : null}
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}

interface FolderPathProps {
  history: DirectoryEntry[];
  currentPath: string | null;
  canGoBack: boolean;
  onBack: () => void;
  onOpenBreadcrumb: (index: number) => void;
}

function FolderPath({
  history,
  currentPath,
  canGoBack,
  onBack,
  onOpenBreadcrumb,
}: FolderPathProps) {
  return (
    <div className="flex min-w-0 flex-col gap-2">
      <div className="flex min-w-0 items-center gap-2">
        <button
          type="button"
          aria-label="Go to parent folder"
          disabled={!canGoBack}
          onClick={onBack}
          className="flex size-11 shrink-0 items-center justify-center rounded-xl border border-zinc-200 text-zinc-700 outline-none hover:bg-zinc-100 focus-visible:ring-2 focus-visible:ring-violet-600 disabled:cursor-not-allowed disabled:opacity-40"
        >
          <CaretLeftIcon size={18} weight="bold" aria-hidden="true" />
        </button>
        <nav
          aria-label="Folder path"
          className="min-w-0 flex-1 overflow-x-auto"
        >
          <ol className="flex min-w-max items-center gap-1 text-sm">
            <li>
              <button
                type="button"
                aria-label="Computer roots"
                onClick={() => onOpenBreadcrumb(-1)}
                className="flex min-h-11 items-center gap-1 rounded-lg px-2 text-zinc-600 outline-none hover:bg-zinc-100 focus-visible:ring-2 focus-visible:ring-violet-600"
              >
                <HouseIcon size={16} aria-hidden="true" />
                Computer
              </button>
            </li>
            {history.map((item, index) => (
              <li key={item.directory_ref} className="flex items-center gap-1">
                <CaretRightIcon
                  size={14}
                  className="text-zinc-400"
                  aria-hidden="true"
                />
                <button
                  type="button"
                  aria-current={
                    index === history.length - 1 ? "location" : undefined
                  }
                  onClick={() => onOpenBreadcrumb(index)}
                  className="min-h-11 max-w-48 truncate rounded-lg px-2 text-zinc-700 outline-none hover:bg-zinc-100 focus-visible:ring-2 focus-visible:ring-violet-600"
                >
                  {item.name}
                </button>
              </li>
            ))}
          </ol>
        </nav>
      </div>
      {currentPath ? (
        <code
          translate="no"
          className="break-all rounded-lg bg-zinc-100 px-3 py-2 text-left text-xs text-zinc-600"
        >
          {currentPath}
        </code>
      ) : null}
    </div>
  );
}

function platformLabel(platform: string) {
  const labels: Record<string, string> = {
    darwin: "macOS",
    linux: "Linux",
    windows: "Windows",
    wsl2: "Windows (WSL2)",
  };
  return labels[platform.toLowerCase()] ?? platform;
}
