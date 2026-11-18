"use client";

import type { ExecutorMachine } from "@/app/api/__generated__/models/executorMachine";
import { useState } from "react";
import { computerOptions } from "./helpers";

interface Args {
  isOpen: boolean;
  machines: ExecutorMachine[];
}

// Kept by the folder dialog, which stays mounted while a failed check puts an
// error in place of the computer list. So the open steps, and a computer that
// connected meanwhile, are still there when the list comes back.
export function useAnotherComputer({ isOpen, machines }: Args) {
  const [isSetupOpen, setSetupOpen] = useState(false);
  const [knownMachineIDs, setKnownMachineIDs] = useState(() =>
    machines.map((machine) => machine.machine_id),
  );
  const [newcomerID, setNewcomerID] = useState<string | null>(null);

  const hasSameMachines =
    machines.length === knownMachineIDs.length &&
    machines.every((machine) => knownMachineIDs.includes(machine.machine_id));
  if (!hasSameMachines) {
    setKnownMachineIDs(machines.map((machine) => machine.machine_id));
    const joined = machines.find(
      (machine) => !knownMachineIDs.includes(machine.machine_id),
    );
    // A first computer isn't another one: the dialog announces that itself.
    if (isOpen && joined && knownMachineIDs.length > 0) {
      setNewcomerID(joined.machine_id);
      setSetupOpen(false);
    }
  }
  if (!isOpen && (isSetupOpen || newcomerID !== null)) {
    setSetupOpen(false);
    setNewcomerID(null);
  }

  const newcomer = computerOptions(machines).find(
    (option) => option.value === newcomerID,
  );

  function toggleSetup() {
    setSetupOpen((wasOpen) => !wasOpen);
  }

  return {
    isSetupOpen,
    toggleSetup,
    connectedNotice: newcomer
      ? `${newcomer.label} connected. Choose it under Computer.`
      : "",
  };
}
