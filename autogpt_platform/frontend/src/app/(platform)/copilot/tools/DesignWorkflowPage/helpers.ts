import { ResponseType } from "@/app/api/__generated__/models/responseType";
import type { WorkflowPageResponse } from "@/app/api/__generated__/models/workflowPageResponse";
import type { ToolUIPart } from "ai";

export const DESIGN_WORKFLOW_PAGE_PART_TYPE = "tool-design_workflow_page";
export const APPLY_MESSAGE = "Apply this page.";
export const DISCARD_MESSAGE = "Discard this page.";

export function parseWorkflowPage(
  part: ToolUIPart,
): WorkflowPageResponse | null {
  const output = toRecord(part.output);
  if (!output || output.type !== ResponseType.workflow_page) return null;
  const page = toRecord(output.page);
  if (!page || !Array.isArray(page.sections)) return null;
  return output as unknown as WorkflowPageResponse;
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
