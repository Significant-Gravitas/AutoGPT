import type { ComputerInfo } from "@/app/api/__generated__/models/computerInfo";
import type { SandboxSummary } from "@/app/api/__generated__/models/sandboxSummary";

export function formatSandboxState(summary: SandboxSummary | null | undefined) {
  if (!summary) return "Not created yet";
  if (summary.state === "running") {
    const since = new Date(summary.started_at);
    return `Running since ${since.toLocaleTimeString([], {
      hour: "numeric",
      minute: "2-digit",
    })}`;
  }
  return "Suspended. Resumes in about a second when needed.";
}

export function formatResources(summary: SandboxSummary | null | undefined) {
  if (!summary) return null;
  const gib = summary.memory_mb / 1024;
  const memory =
    gib >= 1
      ? `${Number.isInteger(gib) ? gib : gib.toFixed(1)} GiB`
      : `${summary.memory_mb} MiB`;
  return `${summary.cpu_count} vCPU · ${memory}`;
}

export function formatScreen(
  computer: Pick<ComputerInfo, "box" | "screen_on">,
) {
  if (!computer.box) {
    return "Screen off. It comes on the first time a task needs a browser or GUI app.";
  }
  return computer.screen_on
    ? "Screen on. Open the desktop to watch or take over."
    : "Screen off. Turning it on happens inside this same machine.";
}

export function describeMount(
  path: string,
  computer: Pick<ComputerInfo, "workspace_path" | "shared_path" | "owner_kind">,
) {
  if (path === computer.workspace_path) {
    return computer.owner_kind === "expert"
      ? "Its own home. Tools, notes and configs live here."
      : "Your workspace.";
  }
  if (path === computer.shared_path) {
    return "Your workspace, shared. Deliverables dropped here reach your desktop.";
  }
  return "Mounted volume.";
}

export function desktopActionLabel(computer: ComputerInfo | null | undefined) {
  if (!computer?.box || !computer.screen_on) return "Turn on screen";
  return computer.box.state === "running" ? "Open desktop" : "Resume desktop";
}
