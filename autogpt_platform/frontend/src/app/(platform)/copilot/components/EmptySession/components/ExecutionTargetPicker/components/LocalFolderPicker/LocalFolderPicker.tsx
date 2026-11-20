"use client";

import type { DirectoryEntry } from "@/app/api/__generated__/models/directoryEntry";
import type { ExecutorMachine } from "@/app/api/__generated__/models/executorMachine";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Select } from "@/components/atoms/Select/Select";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  ArrowLeft01Icon,
  ArrowRight01Icon,
  Folder01Icon,
  Home01Icon,
  Alert02Icon,
} from "@hugeicons/core-free-icons";
import { useEffect, useState } from "react";
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
  const canSelectDirectory =
    !!machine &&
    !!browser.directory?.current &&
    browser.directory.connection_id === machine.connection_id &&
    !browser.isLoading &&
    !browser.isLoadingMore &&
    !browser.error &&
    !isMachinesError &&
    !isLoadingMachines;
  const [knownMachineCount, setKnownMachineCount] = useState(machines.length);
  const [machineAnnouncement, setMachineAnnouncement] = useState("");
  if (machines.length !== knownMachineCount) {
    setKnownMachineCount(machines.length);
    if (isOpen && knownMachineCount === 0 && machines.length > 0) {
      setMachineAnnouncement(
        `${machines[0].display_name} connected. Choose a computer and folder.`,
      );
    }
  }

  useEffect(
    function focusNewConnection() {
      if (!machineAnnouncement) return;
      const frame = requestAnimationFrame(() => {
        document.getElementById("local-executor-machine")?.focus();
      });
      return () => cancelAnimationFrame(frame);
    },
    [machineAnnouncement],
  );

  function handleUseFolder() {
    if (!canSelectDirectory || !machine || !browser.directory?.current) return;
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
              className="flex flex-col items-center gap-3 rounded-2xl border border-error/20 bg-error-surface p-6 text-center text-error-foreground"
            >
              <Icon icon={Alert02Icon} size={24} aria-hidden="true" />
              <Text variant="body-medium" tone="inherit">
                Could Not Check Your Computers
              </Text>
              <Text variant="small" tone="inherit">
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
                className="min-h-0 flex-1 rounded-2xl border border-border-base bg-surface"
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
                    <span className="flex text-warning-foreground">
                      <Icon icon={Alert02Icon} size={24} aria-hidden="true" />
                    </span>
                    <Text variant="small-medium" tone="primary">
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
                    className="flex min-h-48 items-center justify-center p-6 text-center text-body text-foreground-tertiary"
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
                          className="flex min-h-11 w-full min-w-0 items-center gap-3 rounded-xl px-3 py-2 text-left text-body text-foreground-emphasis outline-none hover:bg-surface-hover focus-visible:ring-2 focus-visible:ring-brand focus-visible:ring-offset-1"
                        >
                          <span className="flex shrink-0 text-brand">
                            <Icon
                              icon={Folder01Icon}
                              size={20}
                              aria-hidden="true"
                            />
                          </span>
                          <span className="min-w-0 flex-1 truncate">
                            {entry.name}
                          </span>
                          <span className="flex shrink-0 text-foreground-subtle">
                            <Icon
                              icon={ArrowRight01Icon}
                              size={16}
                              aria-hidden="true"
                            />
                          </span>
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
                <Text role="status" variant="small" tone="warning">
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
              disabled={!canSelectDirectory}
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
          className="flex size-11 shrink-0 items-center justify-center rounded-xl border border-border-base text-foreground-body outline-none hover:bg-surface-hover focus-visible:ring-2 focus-visible:ring-brand disabled:cursor-not-allowed disabled:opacity-40"
        >
          <Icon icon={ArrowLeft01Icon} size={18} aria-hidden="true" />
        </button>
        <nav
          aria-label="Folder path"
          className="min-w-0 flex-1 overflow-x-auto"
        >
          <ol className="flex min-w-max items-center gap-1 text-body">
            <li>
              <button
                type="button"
                aria-label="Computer roots"
                onClick={() => onOpenBreadcrumb(-1)}
                className="flex min-h-11 items-center gap-1 rounded-lg px-2 text-foreground-secondary outline-none hover:bg-surface-hover focus-visible:ring-2 focus-visible:ring-brand"
              >
                <Icon icon={Home01Icon} size={16} aria-hidden="true" />
                Computer
              </button>
            </li>
            {history.map((item, index) => (
              <li key={item.directory_ref} className="flex items-center gap-1">
                <span className="flex text-foreground-subtle">
                  <Icon icon={ArrowRight01Icon} size={14} aria-hidden="true" />
                </span>
                <button
                  type="button"
                  aria-current={
                    index === history.length - 1 ? "location" : undefined
                  }
                  onClick={() => onOpenBreadcrumb(index)}
                  className="min-h-11 max-w-48 truncate rounded-lg px-2 text-foreground-body outline-none hover:bg-surface-hover focus-visible:ring-2 focus-visible:ring-brand"
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
          className="rounded-lg bg-surface-hover px-3 py-2 text-left text-caption break-all text-foreground-secondary"
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
