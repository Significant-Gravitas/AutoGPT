import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import type { UserDailyCost } from "@/app/api/__generated__/models/userDailyCost";

export interface AgentLookupEntry {
  libraryAgentId: string;
  name: string;
  imageUrl?: string | null;
}

function toIsoDate(d: Date): string {
  const yyyy = d.getUTCFullYear();
  const mm = String(d.getUTCMonth() + 1).padStart(2, "0");
  const dd = String(d.getUTCDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
}

function parseIsoDate(value: string | Date): Date | null {
  const iso = value instanceof Date ? toIsoDate(value) : value;
  const [year, month, day] = iso.split("-").map((s) => Number(s));
  if (!year || !month || !day) return null;
  return new Date(Date.UTC(year, month - 1, day));
}

/**
 * Server returns only days with at least one execution; the chart would
 * otherwise have ghost-gaps that read as "those days fell outside the
 * window" rather than "zero spend on those days." Walk from min→max date
 * and fill missing UTC days with zero buckets.
 *
 * Note: the generated `UserDailyCost.date` type is `Date`, but JSON-over-the-
 * wire delivers a `YYYY-MM-DD` string. We normalise to ISO strings here so
 * downstream consumers can treat all entries uniformly.
 */
export function fillDailyGaps(daily: UserDailyCost[]): UserDailyCost[] {
  if (daily.length === 0) return daily;
  const first = parseIsoDate(daily[0].date);
  const last = parseIsoDate(daily[daily.length - 1].date);
  if (!first || !last) return daily;

  const byDate = new Map<string, UserDailyCost>();
  for (const entry of daily) {
    const iso = entry.date instanceof Date ? toIsoDate(entry.date) : entry.date;
    byDate.set(iso, { ...entry, date: iso as unknown as Date });
  }

  const filled: UserDailyCost[] = [];
  for (let t = first.getTime(); t <= last.getTime(); t += 86_400_000) {
    const iso = toIsoDate(new Date(t));
    filled.push(
      byDate.get(iso) ?? {
        date: iso as unknown as Date,
        cost_cents: 0,
        run_count: 0,
      },
    );
  }
  return filled;
}

export function buildAgentLookup(
  agents: LibraryAgent[],
): Map<string, AgentLookupEntry> {
  const map = new Map<string, AgentLookupEntry>();
  for (const agent of agents) {
    map.set(agent.graph_id, {
      libraryAgentId: agent.id,
      name: agent.name,
      imageUrl: agent.image_url,
    });
  }
  return map;
}

export function formatRelativeDate(input: string | Date): string {
  const date = input instanceof Date ? input : new Date(input);
  const diffMs = Date.now() - date.getTime();
  const minutes = Math.round(diffMs / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  if (days < 30) return `${days}d ago`;
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

// Orval types `date: Date` because the OpenAPI schema declares format=date,
// but JSON serialisation actually delivers an ISO string like "2026-05-10"
// (no Date reviver runs on the client). Accept both at runtime.
export function formatShortDate(input: string | Date): string {
  const iso = input instanceof Date ? toIsoDate(input) : input;
  const [year, month, day] = iso.split("-").map((s) => Number(s));
  if (!year || !month || !day) return iso;
  const date = new Date(Date.UTC(year, month - 1, day));
  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}
