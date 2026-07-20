"use client";

import { useGetExperimentalListLocalExecutors } from "@/app/api/__generated__/endpoints/copilot/copilot";
import type { ExecutorMachine } from "@/app/api/__generated__/models/executorMachine";
import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { useEffect } from "react";

const EXECUTOR_POLL_INTERVAL_MS = 3_000;

function localTargetForMachine(machine: ExecutorMachine) {
  return {
    kind: "local" as const,
    machineID: machine.machine_id,
    machineLabel: machine.display_name,
    connectionID: machine.connection_id,
    browseID: null,
    directoryRef: null,
    displayPath: null,
  };
}

export function useExecutionTargetPicker() {
  const target = useCopilotUIStore((state) => state.newChatExecutionTarget);
  const setTarget = useCopilotUIStore(
    (state) => state.setNewChatExecutionTarget,
  );
  const isOpen = useCopilotUIStore(
    (state) => state.isExecutionTargetPickerOpen,
  );
  const setOpen = useCopilotUIStore(
    (state) => state.setExecutionTargetPickerOpen,
  );
  const error = useCopilotUIStore((state) => state.executionTargetError);
  const setError = useCopilotUIStore((state) => state.setExecutionTargetError);

  const machinesQuery = useGetExperimentalListLocalExecutors({
    query: {
      enabled: target.kind === "local" || isOpen,
      refetchInterval: EXECUTOR_POLL_INTERVAL_MS,
      staleTime: 2_000,
    },
  });

  const machines =
    machinesQuery.data?.status === 200 ? machinesQuery.data.data.executors : [];

  useEffect(
    function reconcileSelectedMachine() {
      if (target.kind !== "local") return;

      if (machines.length === 0) {
        if (target.machineID || target.connectionID || target.directoryRef) {
          setTarget({
            kind: "local",
            machineID: null,
            machineLabel: null,
            connectionID: null,
            browseID: null,
            directoryRef: null,
            displayPath: null,
          });
          setError(
            `${target.machineLabel ?? "The selected computer"} is offline. Reconnect it before starting this chat.`,
          );
        }
        return;
      }

      if (!target.machineID) {
        setTarget(localTargetForMachine(machines[0]));
        return;
      }

      const connectedMachine = machines.find(
        (machine) => machine.machine_id === target.machineID,
      );
      if (!connectedMachine) {
        const replacement = machines[0];
        setTarget(localTargetForMachine(replacement));
        setError(
          `${target.machineLabel ?? "The selected computer"} is offline. Choose a folder on ${replacement.display_name} instead.`,
        );
        return;
      }

      if (connectedMachine.connection_id !== target.connectionID) {
        setTarget(localTargetForMachine(connectedMachine));
        setError("Your Local PC reconnected. Choose the folder again.");
      }
    },
    [machines, setError, setTarget, target],
  );

  function selectCloud() {
    setTarget({ kind: "cloud" });
    setError(null);
    setOpen(false);
  }

  function selectLocal() {
    setError(null);
    if (target.kind !== "local") {
      const firstMachine = machines[0];
      setTarget(
        firstMachine
          ? localTargetForMachine(firstMachine)
          : {
              kind: "local",
              machineID: null,
              machineLabel: null,
              connectionID: null,
              browseID: null,
              directoryRef: null,
              displayPath: null,
            },
      );
    }
    setOpen(true);
  }

  function selectMachine(machineID: string) {
    const machine = machines.find((item) => item.machine_id === machineID);
    if (!machine) return;
    setTarget(localTargetForMachine(machine));
    setError(null);
  }

  function selectDirectory(
    machine: ExecutorMachine,
    browseID: string,
    directoryRef: string,
    displayPath: string,
  ) {
    setTarget({
      kind: "local",
      machineID: machine.machine_id,
      machineLabel: machine.display_name,
      connectionID: machine.connection_id,
      browseID,
      directoryRef,
      displayPath,
    });
    setError(null);
    setOpen(false);
  }

  return {
    target,
    isOpen,
    setOpen,
    error,
    setError,
    machines,
    isLoadingMachines: machinesQuery.isLoading,
    isRefreshingMachines: machinesQuery.isFetching,
    isMachinesError: machinesQuery.isError,
    retryMachines: machinesQuery.refetch,
    selectCloud,
    selectLocal,
    selectMachine,
    selectDirectory,
  };
}
