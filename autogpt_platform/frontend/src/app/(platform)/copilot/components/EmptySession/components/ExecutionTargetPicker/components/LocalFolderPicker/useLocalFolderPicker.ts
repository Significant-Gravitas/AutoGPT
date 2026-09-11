"use client";

import { postExperimentalListLocalExecutorDirectories } from "@/app/api/__generated__/endpoints/copilot/copilot";
import type { DirectoryEntry } from "@/app/api/__generated__/models/directoryEntry";
import type { DirectoryListRequest } from "@/app/api/__generated__/models/directoryListRequest";
import type { DirectoryListResponse } from "@/app/api/__generated__/models/directoryListResponse";
import type { ExecutorMachine } from "@/app/api/__generated__/models/executorMachine";
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
  const [lastTarget, setLastTarget] = useState<NavigationTarget>({
    directoryRef: null,
    history: [],
  });
  const requestIDRef = useRef(0);

  useEffect(
    function loadDirectoryRoots() {
      if (!isOpen || !machine) {
        requestIDRef.current += 1;
        setDirectory(null);
        setHistory([]);
        setError(null);
        return;
      }

      const requestID = requestIDRef.current + 1;
      requestIDRef.current = requestID;
      setIsLoading(true);
      setError(null);
      setLastTarget({ directoryRef: null, history: [] });

      void browseLocalExecutorDirectory(machine.machine_id, {
        expected_connection_id: machine.connection_id,
        browse_id: null,
        directory_ref: null,
      })
        .then((response) => {
          if (requestIDRef.current !== requestID) return;
          if (response.connection_id !== machine.connection_id) {
            onStale("Your Local PC reconnected. Choose the folder again.");
            setError("The computer connection changed. Refresh and try again.");
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
    [isOpen, machine?.connection_id, machine?.machine_id, onStale],
  );

  function handleRequestError(requestError: unknown) {
    if (requestError instanceof ApiError && requestError.status === 409) {
      const message =
        "The computer or folder changed while you were browsing. Refresh and choose it again.";
      setDirectory(null);
      setHistory([]);
      setLastTarget({ directoryRef: null, history: [] });
      setError(message);
      onStale(message);
      return;
    }
    setError("Could not load folders from this computer. Try again.");
  }

  async function loadDirectory(target: NavigationTarget) {
    if (!machine) return;
    const requestID = requestIDRef.current + 1;
    requestIDRef.current = requestID;
    setIsLoading(true);
    setError(null);
    setLastTarget(target);
    try {
      const response = await browseLocalExecutorDirectory(machine.machine_id, {
        expected_connection_id: machine.connection_id,
        browse_id: target.directoryRef ? (directory?.browse_id ?? null) : null,
        directory_ref: target.directoryRef,
      });
      if (requestIDRef.current !== requestID) return;
      if (response.connection_id !== machine.connection_id) {
        const message = "Your Local PC reconnected. Choose the folder again.";
        setError(message);
        onStale(message);
        return;
      }
      setDirectory(response);
      setHistory(target.history);
    } catch (requestError: unknown) {
      if (requestIDRef.current !== requestID) return;
      handleRequestError(requestError);
    } finally {
      if (requestIDRef.current === requestID) setIsLoading(false);
    }
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

  function retry() {
    void loadDirectory(lastTarget);
  }

  async function loadMore() {
    if (!machine || !directory?.current || !directory.next_cursor) return;
    const currentDirectory = directory;
    const current = directory.current;
    const requestID = requestIDRef.current + 1;
    requestIDRef.current = requestID;
    setIsLoadingMore(true);
    setError(null);
    try {
      const response = await browseLocalExecutorDirectory(machine.machine_id, {
        expected_connection_id: machine.connection_id,
        browse_id: currentDirectory.browse_id,
        directory_ref: current.directory_ref,
        cursor: currentDirectory.next_cursor,
      });
      if (requestIDRef.current !== requestID) return;
      if (
        response.connection_id !== machine.connection_id ||
        response.browse_id !== currentDirectory.browse_id ||
        response.current?.directory_ref !== current.directory_ref
      ) {
        const message = "Your Local PC folder view changed. Choose it again.";
        setError(message);
        onStale(message);
        return;
      }
      const entries = new Map(
        currentDirectory.entries.map((entry) => [entry.directory_ref, entry]),
      );
      for (const entry of response.entries) {
        entries.set(entry.directory_ref, entry);
      }
      setDirectory({ ...response, entries: Array.from(entries.values()) });
    } catch (requestError: unknown) {
      if (requestIDRef.current !== requestID) return;
      handleRequestError(requestError);
    } finally {
      if (requestIDRef.current === requestID) setIsLoadingMore(false);
    }
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
