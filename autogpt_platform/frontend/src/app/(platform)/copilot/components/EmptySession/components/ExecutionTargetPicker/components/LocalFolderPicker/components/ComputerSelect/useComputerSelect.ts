"use client";

import type { ExecutorMachine } from "@/app/api/__generated__/models/executorMachine";
import { useState } from "react";

export function useComputerSelect(machines: ExecutorMachine[]) {
  const [isSetupOpen, setSetupOpen] = useState(false);
  const [knownMachineIDs, setKnownMachineIDs] = useState(() =>
    machines.map((machine) => machine.machine_id),
  );
  const [connectedNotice, setConnectedNotice] = useState("");

  const hasSameMachines =
    machines.length === knownMachineIDs.length &&
    machines.every((machine) => knownMachineIDs.includes(machine.machine_id));
  if (!hasSameMachines) {
    setKnownMachineIDs(machines.map((machine) => machine.machine_id));
    const newcomer = machines.find(
      (machine) => !knownMachineIDs.includes(machine.machine_id),
    );
    if (newcomer) {
      setConnectedNotice(
        `${newcomer.display_name} connected. Choose it under Computer.`,
      );
      setSetupOpen(false);
    }
  }

  function toggleSetup() {
    setSetupOpen((isOpen) => !isOpen);
  }

  return { isSetupOpen, toggleSetup, connectedNotice };
}
