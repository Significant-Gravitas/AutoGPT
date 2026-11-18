import type { ExecutorMachine } from "@/app/api/__generated__/models/executorMachine";

const SHORT_ID_LENGTH = 8;

const PLATFORM_LABELS: Record<string, string> = {
  darwin: "macOS",
  linux: "Linux",
  windows: "Windows",
  wsl2: "Windows (WSL2)",
};

function computerLabel(machine: ExecutorMachine) {
  const platform =
    PLATFORM_LABELS[machine.platform.toLowerCase()] ?? machine.platform;
  return `${machine.display_name} · ${platform}`;
}

export function hasComputersWithOneName(machines: ExecutorMachine[]) {
  return new Set(machines.map(computerLabel)).size !== machines.length;
}

// Computers that would show the same name also show the start of their ID,
// which `autogpt-shim start` prints on each of them.
export function computerOptions(machines: ExecutorMachine[]) {
  const labels = machines.map(computerLabel);
  return machines.map((machine, index) => {
    const label = labels[index];
    const isShared = labels.indexOf(label) !== labels.lastIndexOf(label);
    return {
      value: machine.machine_id,
      label: isShared
        ? `${label} · ${machine.machine_id.slice(0, SHORT_ID_LENGTH)}`
        : label,
    };
  });
}
