import * as Sentry from "@sentry/nextjs";
import type { GraphID, GraphExecutionID } from "@/lib/autogpt-server-api/types";

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

function isUuid(value: string): boolean {
  return UUID_RE.test(value);
}

export function parseGraphExecutionID(
  raw: string | null | undefined,
): GraphExecutionID | null {
  if (!raw) return null;
  const trimmed = raw.trim();
  if (!trimmed) return null;
  if (!isUuid(trimmed)) {
    Sentry.captureMessage("Invalid GraphExecutionID rejected", {
      level: "warning",
      tags: { invalid_execution_id: "true" },
      extra: { raw: trimmed.slice(0, 64) },
    });
    return null;
  }
  return trimmed as GraphExecutionID;
}

export function parseGraphID(raw: string | null | undefined): GraphID | null {
  if (!raw) return null;
  const trimmed = raw.trim();
  if (!trimmed) return null;
  if (!isUuid(trimmed)) {
    Sentry.captureMessage("Invalid GraphID rejected", {
      level: "warning",
      tags: { invalid_graph_id: "true" },
      extra: { raw: trimmed.slice(0, 64) },
    });
    return null;
  }
  return trimmed as GraphID;
}

export function isValidGraphExecutionID(
  raw: string | null | undefined,
): raw is GraphExecutionID {
  return parseGraphExecutionID(raw) !== null;
}

export function isValidGraphID(
  raw: string | null | undefined,
): raw is GraphID {
  return parseGraphID(raw) !== null;
}
