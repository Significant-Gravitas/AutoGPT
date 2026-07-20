import type { RecordingStartRequestChannelsItem } from "@/app/api/__generated__/models/recordingStartRequestChannelsItem";
import type { RecordingStartRequestInterpretationRoute } from "@/app/api/__generated__/models/recordingStartRequestInterpretationRoute";
import type { TrajectoryStep } from "@/app/api/__generated__/models/trajectoryStep";

export interface CapturedStep {
  seq: number;
  timestamp: number;
  actor: string;
  action: string;
  label?: string | null;
  screenshotRef?: string | null;
  cursor?: [number, number] | null;
  activeApp?: string | null;
  activeWindow?: string | null;
  enrichment: {
    kind: string;
    selectors: { strategy: string; value: string }[];
    axPath?: string | null;
    role?: string | null;
    label?: string | null;
    url?: string | null;
  };
  narration?: string | null;
  value?: string | null;
  valueType?: string | null;
  isParameter?: boolean | null;
  outcome: string;
  redacted?: boolean;
}

function displayValue(raw: unknown): string | null {
  if (raw === null || raw === undefined) return null;
  if (typeof raw === "string") return raw;
  if (typeof raw === "number" || typeof raw === "boolean") return String(raw);
  return JSON.stringify(raw) ?? String(raw);
}

const ROUTE_PRIORITY: RecordingStartRequestInterpretationRoute[] = [
  "extract_then_cloud",
  "local_vlm",
  "screenshots_to_cloud",
];

const CHANNEL_PRIORITY: RecordingStartRequestChannelsItem[] = [
  "floor",
  "browser",
  "desktop_ax",
];

export interface RecordingSettings {
  interpretationRoute: RecordingStartRequestInterpretationRoute;
  channels: RecordingStartRequestChannelsItem[];
}

export function selectRecordingSettings(
  routes: RecordingStartRequestInterpretationRoute[] | null | undefined,
  channels: RecordingStartRequestChannelsItem[] | null | undefined,
): RecordingSettings | null {
  const interpretationRoute = ROUTE_PRIORITY.find((route) =>
    routes?.includes(route),
  );
  const supportedChannels = CHANNEL_PRIORITY.filter((channel) =>
    channels?.includes(channel),
  );
  if (!interpretationRoute || supportedChannels.length === 0) return null;
  return { interpretationRoute, channels: supportedChannels };
}

export function toCapturedStep(step: TrajectoryStep): CapturedStep {
  const cursor =
    step.cursor?.length === 2
      ? ([step.cursor[0], step.cursor[1]] as [number, number])
      : null;
  const selectors = (step.enrichment?.selectors ?? []).flatMap((selector) => {
    const strategy = selector.strategy;
    const value = selector.value;
    return typeof strategy === "string" && typeof value === "string"
      ? [{ strategy, value }]
      : [];
  });

  return {
    seq: step.seq ?? 0,
    timestamp: step.ts ?? 0,
    actor: step.actor ?? "human",
    action: step.action ?? "",
    label:
      step.enrichment?.label ?? step.active_window ?? step.narration ?? null,
    screenshotRef: step.screenshot_ref,
    cursor,
    activeApp: step.active_app,
    activeWindow: step.active_window,
    enrichment: {
      kind: step.enrichment?.kind ?? "none",
      selectors,
      axPath: step.enrichment?.ax_path,
      role: step.enrichment?.role,
      label: step.enrichment?.label,
      url: step.enrichment?.url,
    },
    narration: step.narration,
    value: displayValue(step.value?.raw),
    valueType: step.value?.type ?? null,
    isParameter: step.value?.is_parameter ?? null,
    outcome: step.outcome ?? "unknown",
    redacted: step.redacted,
  };
}
