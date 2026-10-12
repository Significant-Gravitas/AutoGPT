import type { DelegationSummaryStatus } from "@/app/api/__generated__/models/delegationSummaryStatus";
import type { UIMessage } from "ai";
import { isSessionOpeningMessage } from "../../../../sentFrom";

type ThreadTag = { label: string; tone: "working" | "done" | "stopped" };

export function getThreadTag(
  status: DelegationSummaryStatus | null,
  from: string,
): ThreadTag | null {
  if (!status) return null;
  if (status === "completed")
    return { label: `Done for ${from}`, tone: "done" };
  if (status === "failed" || status === "cancelled")
    return { label: "Stopped", tone: "stopped" };
  return { label: `Working for ${from}`, tone: "working" };
}

export function formatUsd(value: number | null | undefined) {
  return value === null || value === undefined ? null : `$${value.toFixed(2)}`;
}

/** "$2.00 cap · $0.12 used", or whichever half is known. */
export function getCapLine(
  capUsd: number | null | undefined,
  costUsd: number | null | undefined,
) {
  const cap = formatUsd(capUsd);
  const used = formatUsd(costUsd);
  if (!cap && !used) return null;
  return [cap && `${cap} cap`, used && `${used} used`]
    .filter(Boolean)
    .join(" · ");
}

export function formatClock(value: Date | string | null | undefined) {
  if (!value) return null;
  return new Date(value).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  });
}

/** The text of the message that opened the thread: the brief it was sent. */
export function getOpeningText(messages: UIMessage[]) {
  const opening = messages.find(isSessionOpeningMessage);
  if (!opening) return null;
  const text = opening.parts
    .flatMap((part) => (part.type === "text" ? [part.text] : []))
    .join("\n")
    .trim();
  return text || null;
}
