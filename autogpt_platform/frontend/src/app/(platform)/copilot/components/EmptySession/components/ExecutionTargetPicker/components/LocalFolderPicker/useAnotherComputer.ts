"use client";

import type { ExecutorMachine } from "@/app/api/__generated__/models/executorMachine";
import { useState } from "react";
import { computerMenu } from "./helpers";

interface Args {
  isOpen: boolean;
  machines: ExecutorMachine[];
}

const NAMES = new Intl.ListFormat("en", { type: "conjunction" });

// Kept by the folder dialog, which stays mounted while a failed check puts an
// error in place of the computer list. So the open steps, and a computer that
// connected meanwhile, are still there when the list comes back. The dialog
// itself starts afresh when the chosen computer changes or reconnects.
export function useAnotherComputer({ isOpen, machines }: Args) {
  const [isSetupOpen, setSetupOpen] = useState(false);
  const [knownMachineIDs, setKnownMachineIDs] = useState(() =>
    machines.map((machine) => machine.machine_id),
  );
  const [newcomerIDs, setNewcomerIDs] = useState<string[]>([]);

  const hasSameMachines =
    machines.length === knownMachineIDs.length &&
    machines.every((machine) => knownMachineIDs.includes(machine.machine_id));
  if (!hasSameMachines) {
    setKnownMachineIDs(machines.map((machine) => machine.machine_id));
    const joined = machines
      .map((machine) => machine.machine_id)
      .filter((machineID) => !knownMachineIDs.includes(machineID));
    // A first computer isn't another one: the dialog announces that itself.
    if (isOpen && joined.length > 0 && knownMachineIDs.length > 0) {
      setNewcomerIDs(joined);
      setSetupOpen(false);
    }
  }
  if (!isOpen && (isSetupOpen || newcomerIDs.length > 0)) {
    setSetupOpen(false);
    setNewcomerIDs([]);
  }

  const newcomers = computerMenu(machines)
    .options.filter((option) => newcomerIDs.includes(option.value))
    .map((option) => option.label);

  function toggleSetup() {
    setSetupOpen((wasOpen) => !wasOpen);
  }

  return {
    isSetupOpen,
    toggleSetup,
    connectedNotice: connectedNotice(newcomers),
  };
}

function connectedNotice(labels: string[]) {
  if (labels.length === 0) return "";
  const which = labels.length === 1 ? "it" : "one";
  return `${NAMES.format(labels)} connected. Choose ${which} under Computer.`;
}
