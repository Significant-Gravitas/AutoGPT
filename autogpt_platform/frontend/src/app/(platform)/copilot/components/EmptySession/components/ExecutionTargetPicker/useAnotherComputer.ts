"use client";

import type { ExecutorMachine } from "@/app/api/__generated__/models/executorMachine";
import { useState } from "react";
import { computerMenu } from "./components/LocalFolderPicker/helpers";

interface Args {
  isOpen: boolean;
  machines: ExecutorMachine[];
  selectedMachineID: string | null;
}

const NAMES = new Intl.ListFormat("en", { type: "conjunction" });

// Kept above the folder dialog. The dialog starts afresh whenever the chosen
// computer changes or reconnects, and a failed check puts an error in place
// of its computer list. The open steps, and a computer that connected
// meanwhile, outlast both.
export function useAnotherComputer({
  isOpen,
  machines,
  selectedMachineID,
}: Args) {
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
      setNewcomerIDs((earlier) => [
        ...earlier.filter((machineID) => !joined.includes(machineID)),
        ...joined,
      ]);
      setSetupOpen(false);
    }
  }
  const hasNoMenu = !isOpen || machines.length === 0;
  if (hasNoMenu && (isSetupOpen || newcomerIDs.length > 0)) {
    setSetupOpen(false);
    setNewcomerIDs([]);
  }
  // Once chosen, a computer is no longer news, whatever is chosen after it.
  if (selectedMachineID && newcomerIDs.includes(selectedMachineID)) {
    // From what is pending, not from what this render began with: a computer
    // that joined in this same render stays in.
    setNewcomerIDs((pending) =>
      pending.filter((machineID) => machineID !== selectedMachineID),
    );
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
