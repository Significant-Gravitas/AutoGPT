import type { PageResponse } from "@/app/api/__generated__/models/pageResponse";
import type { ResolvedPage } from "@/app/api/__generated__/models/resolvedPage";
import { ResponseType } from "@/app/api/__generated__/models/responseType";
import type { ToolUIPart } from "ai";

export const DESIGN_PAGE_PART_TYPE = "tool-design_page";
export const APPLY_MESSAGE = "Apply this page.";
export const DISCARD_MESSAGE = "Discard this page.";

type PageOutput = PageResponse & { page: ResolvedPage };

/** The page a design_page call drew, or null for a catalogue or a read. */
export function parsePageOutput(part: ToolUIPart): PageOutput | null {
  const output = toRecord(part.output);
  if (!output || output.type !== ResponseType.page) return null;
  const page = toRecord(output.page);
  if (!page || !Array.isArray(page.sections)) return null;
  return output as unknown as PageOutput;
}

/** Whether the stored page moved on since this preview was drawn. */
export function isOutdated(
  page: ResolvedPage,
  stored: { id: string; revision: number }[] | undefined,
) {
  if (!page.id || page.revision == null || !stored) return false;
  const current = stored.find((p) => p.id === page.id);
  return Boolean(current && current.revision > page.revision);
}

function toRecord(value: unknown): Record<string, unknown> | null {
  if (typeof value === "string") {
    try {
      return toRecord(JSON.parse(value) as unknown);
    } catch {
      return null;
    }
  }
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}
