"use client";

import { postExperimentalListLocalExecutorDirectories } from "@/app/api/__generated__/endpoints/copilot/copilot";
import type { DirectoryEntry } from "@/app/api/__generated__/models/directoryEntry";
import type { DirectoryListRequest } from "@/app/api/__generated__/models/directoryListRequest";
import type { DirectoryListResponse } from "@/app/api/__generated__/models/directoryListResponse";
import type { ExecutorMachine } from "@/app/api/__generated__/models/executorMachine";
import { useStableCallback } from "@/hooks/useStableCallback";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { useEffect, useRef, useState } from "react";

interface Props {
  isOpen: boolean;
  machine: ExecutorMachine | null;
  onStale: (message: string) => void;
}

interface NavigationTarget {
  directoryRef: string | null;
  history: DirectoryEntry[];
}

export function useLocalFolderPicker({ isOpen, machine, onStale }: Props) {
  const [directory, setDirectory] = useState<DirectoryListResponse | null>(
    null,
  );
  const [history, setHistory] = useState<DirectoryEntry[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // A failed "load more" keeps the loaded pages and their cursor, so a retry
  // asks for that page again instead of reloading the folder.
  const [pageFailed, setPageFailed] = useState(false);
  const [lastTarget, setLastTarget] = useState<NavigationTarget>({
    directoryRef: null,
    history: [],
  });
  const requestIDRef = useRef(0);
  const machineID = machine?.machine_id ?? null;
  const connectionID = machine?.connection_id ?? null;

  // A different machine, connection or open state starts the browser over:
  // reset while rendering, so no frame shows the previous machine's folders.
  const browseKey =
    isOpen && machineID && connectionID
      ? `${machineID}\n${connectionID}`
      : null;
  const [shownBrowseKey, setShownBrowseKey] = useState<string | null>(null);
  if (browseKey !== shownBrowseKey) {
    setShownBrowseKey(browseKey);
    setDirectory(null);
    setHistory([]);
    setError(null);
    setIsLoading(browseKey !== null);
    setIsLoadingMore(false);
    setPageFailed(false);
    setLastTarget({ directoryRef: null, history: [] });
  }

  const invalidateDirectory = useStableCallback((message: string) => {
    setDirectory(null);
    setHistory([]);
    setLastTarget({ directoryRef: null, history: [] });
    setIsLoadingMore(false);
    setError(message);
    onStale(message);
  });

  const handleRequestError = useStableCallback((requestError: unknown) => {
    if (requestError instanceof ApiError && requestError.status === 409) {
      const message =
        "The computer or folder changed while you were browsing. Refresh and choose it again.";
      invalidateDirectory(message);
      return;
    }
    setError("Could not load folders from this computer. Try again.");
  });

  useEffect(
    function loadDirectoryRoots() {
      if (!browseKey || !machineID || !connectionID) {
        requestIDRef.current += 1;
        return;
      }

      const requestID = requestIDRef.current + 1;
      requestIDRef.current = requestID;

      void browseLocalExecutorDirectory(machineID, {
        expected_connection_id: connectionID,
        browse_id: null,
        directory_ref: null,
      })
        .then((response) => {
          if (requestIDRef.current !== requestID) return;
          if (response.connection_id !== connectionID) {
            invalidateDirectory(
              "Your Local PC reconnected. Choose the folder again.",
            );
            return;
          }
          setDirectory(response);
          setHistory(response.current ? [response.current] : []);
        })
        .catch((requestError: unknown) => {
          if (requestIDRef.current !== requestID) return;
          handleRequestError(requestError);
        })
        .finally(() => {
          if (requestIDRef.current === requestID) setIsLoading(false);
        });

      return () => {
        if (requestIDRef.current === requestID) {
          requestIDRef.current += 1;
        }
      };
    },
    [
      browseKey,
      connectionID,
      handleRequestError,
      invalidateDirectory,
      machineID,
    ],
  );

  function loadDirectory(target: NavigationTarget) {
    if (!machine) return;
    const connectionID = machine.connection_id;
    const requestID = requestIDRef.current + 1;
    requestIDRef.current = requestID;
    setIsLoading(true);
    setIsLoadingMore(false);
    setPageFailed(false);
    setError(null);
    setLastTarget(target);
    const browseID = target.directoryRef
      ? (directory?.browse_id ?? null)
      : null;
    void browseLocalExecutorDirectory(machine.machine_id, {
      expected_connection_id: connectionID,
      browse_id: browseID,
      directory_ref: target.directoryRef,
    })
      .then((response) => {
        if (requestIDRef.current !== requestID) return;
        if (response.connection_id !== connectionID) {
          invalidateDirectory(
            "Your Local PC reconnected. Choose the folder again.",
          );
          return;
        }
        if (
          target.directoryRef &&
          (response.browse_id !== browseID ||
            response.current?.directory_ref !== target.directoryRef)
        ) {
          invalidateDirectory(
            "Your Local PC folder view changed. Choose it again.",
          );
          return;
        }
        setDirectory(response);
        setHistory(target.history);
      })
      .catch((requestError: unknown) => {
        if (requestIDRef.current !== requestID) return;
        handleRequestError(requestError);
      })
      .finally(() => {
        if (requestIDRef.current === requestID) setIsLoading(false);
      });
  }

  function openDirectory(entry: DirectoryEntry) {
    void loadDirectory({
      directoryRef: entry.directory_ref,
      history: [...history, entry],
    });
  }

  function openParent() {
    if (!directory) return;
    void loadDirectory({
      directoryRef: directory.parent_ref ?? null,
      history: history.slice(0, -1),
    });
  }

  function openBreadcrumb(index: number) {
    const item = history[index];
    if (!item) {
      void loadDirectory({ directoryRef: null, history: [] });
      return;
    }
    void loadDirectory({
      directoryRef: item.directory_ref,
      history: history.slice(0, index + 1),
    });
  }

  function loadMore() {
    if (!machine || !directory?.current || !directory.next_cursor) return;
    const connectionID = machine.connection_id;
    const currentDirectory = directory;
    const current = directory.current;
    const requestID = requestIDRef.current + 1;
    requestIDRef.current = requestID;
    setIsLoadingMore(true);
    setPageFailed(false);
    setError(null);
    void browseLocalExecutorDirectory(machine.machine_id, {
      expected_connection_id: connectionID,
      browse_id: currentDirectory.browse_id,
      directory_ref: current.directory_ref,
      cursor: currentDirectory.next_cursor,
    })
      .then((response) => {
        if (requestIDRef.current !== requestID) return;
        if (
          response.connection_id !== connectionID ||
          response.browse_id !== currentDirectory.browse_id ||
          response.current?.directory_ref !== current.directory_ref
        ) {
          invalidateDirectory(
            "Your Local PC folder view changed. Choose it again.",
          );
          return;
        }
        setDirectory({
          ...response,
          entries: mergeEntries(currentDirectory.entries, response.entries),
        });
      })
      .catch((requestError: unknown) => {
        if (requestIDRef.current !== requestID) return;
        setPageFailed(true);
        handleRequestError(requestError);
      })
      .finally(() => {
        if (requestIDRef.current === requestID) setIsLoadingMore(false);
      });
  }

  function retry() {
    if (pageFailed && directory?.next_cursor) {
      loadMore();
      return;
    }
    void loadDirectory(lastTarget);
  }

  return {
    directory,
    history,
    isLoading,
    isLoadingMore,
    error,
    openDirectory,
    openParent,
    openBreadcrumb,
    retry,
    loadMore,
  };
}

function mergeEntries(loaded: DirectoryEntry[], page: DirectoryEntry[]) {
  const entries = new Map(loaded.map((entry) => [entry.directory_ref, entry]));
  for (const entry of page) entries.set(entry.directory_ref, entry);
  return Array.from(entries.values());
}

async function browseLocalExecutorDirectory(
  machineID: string,
  request: DirectoryListRequest,
) {
  const response = await postExperimentalListLocalExecutorDirectories(
    machineID,
    request,
  );
  if (response.status !== 200) {
    throw new Error("The Local PC executor returned an unexpected response");
  }
  return response.data;
}
