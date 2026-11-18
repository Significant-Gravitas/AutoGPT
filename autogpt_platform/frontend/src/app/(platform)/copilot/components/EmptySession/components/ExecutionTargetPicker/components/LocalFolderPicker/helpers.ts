import type { ExecutorMachine } from "@/app/api/__generated__/models/executorMachine";

const SHORT_ID_LENGTH = 8;

const PLATFORM_LABELS: Record<string, string> = {
  darwin: "macOS",
  linux: "Linux",
  windows: "Windows",
  wsl2: "Windows (WSL2)",
};

// Computers that would show the same name also show the start of their ID,
// which `autogpt-shim start` prints on each of them.
export function computerMenu(machines: ExecutorMachine[]) {
  const names = machines.map(computerName);
  const options = machines.map((machine, index) => {
    const name = names[index];
    const sharing = machines.filter((_, other) => names[other] === name);
    if (sharing.length === 1) return { value: machine.machine_id, label: name };
    const length = tellingIDLength(sharing.map((item) => item.machine_id));
    return {
      value: machine.machine_id,
      label: `${name} · ${machine.machine_id.slice(0, length)}`,
    };
  });
  return { options, hasSharedNames: new Set(names).size !== names.length };
}

function computerName(machine: ExecutorMachine) {
  const platform =
    PLATFORM_LABELS[machine.platform.toLowerCase()] ?? machine.platform;
  return `${machine.display_name} · ${platform}`;
}

// How much of their IDs tells these computers apart: eight characters when
// that is enough, more when the IDs start alike.
function tellingIDLength(machineIDs: string[]) {
  const longest = Math.max(...machineIDs.map((id) => id.length));
  for (let length = SHORT_ID_LENGTH; length < longest; length += 1) {
    const starts = new Set(machineIDs.map((id) => id.slice(0, length)));
    if (starts.size === machineIDs.length) return length;
  }
  return longest;
}
