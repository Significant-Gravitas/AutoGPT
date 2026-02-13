import {
  CheckCircleIcon,
  LightbulbIcon,
  MagnifyingGlassIcon,
  PlusCircleIcon,
} from "@phosphor-icons/react";
import type { ToolUIPart } from "ai";

/* ------------------------------------------------------------------ */
/*  Types (local until API client is regenerated)                      */
/* ------------------------------------------------------------------ */

interface FeatureRequestInfo {
  id: string;
  identifier: string;
  title: string;
  description?: string | null;
}

export interface FeatureRequestSearchResponse {
  type: "feature_request_search";
  message: string;
  results: FeatureRequestInfo[];
  count: number;
  query: string;
}

export interface FeatureRequestCreatedResponse {
  type: "feature_request_created";
  message: string;
  issue_id: string;
  issue_identifier: string;
  issue_title: string;
  issue_url: string;
  is_new_issue: boolean;
  customer_name: string;
}

interface NoResultsResponse {
  type: "no_results";
  message: string;
  suggestions?: string[];
}

interface ErrorResponse {
  type: "error";
  message: string;
  error?: string;
}

export type FeatureRequestOutput =
  | FeatureRequestSearchResponse
  | FeatureRequestCreatedResponse
  | NoResultsResponse
  | ErrorResponse;

export type FeatureRequestToolType =
  | "tool-search_feature_requests"
  | "tool-create_feature_request"
  | string;

/* ------------------------------------------------------------------ */
/*  Output parsing                                                     */
/* ------------------------------------------------------------------ */

function parseOutput(output: unknown): FeatureRequestOutput | null {
  if (!output) return null;
  if (typeof output === "string") {
    const trimmed = output.trim();
    if (!trimmed) return null;
    try {
      return parseOutput(JSON.parse(trimmed) as unknown);
    } catch {
      return null;
    }
  }
  if (typeof output === "object") {
    const type = (output as { type?: unknown }).type;
    if (
      type === "feature_request_search" ||
      type === "feature_request_created" ||
      type === "no_results" ||
      type === "error"
    ) {
      return output as FeatureRequestOutput;
    }
    // Fallback structural checks
    if ("results" in output && "query" in output)
      return output as FeatureRequestSearchResponse;
    if ("issue_identifier" in output)
      return output as FeatureRequestCreatedResponse;
    if ("suggestions" in output && !("error" in output))
      return output as NoResultsResponse;
    if ("error" in output || "details" in output)
      return output as ErrorResponse;
  }
  return null;
}

export function getFeatureRequestOutput(
  part: unknown,
): FeatureRequestOutput | null {
  if (!part || typeof part !== "object") return null;
  return parseOutput((part as { output?: unknown }).output);
}

/* ------------------------------------------------------------------ */
/*  Type guards                                                        */
/* ------------------------------------------------------------------ */

export function isSearchResultsOutput(
  output: FeatureRequestOutput,
): output is FeatureRequestSearchResponse {
  return (
    output.type === "feature_request_search" ||
    ("results" in output && "query" in output)
  );
}

export function isCreatedOutput(
  output: FeatureRequestOutput,
): output is FeatureRequestCreatedResponse {
  return (
    output.type === "feature_request_created" || "issue_identifier" in output
  );
}

export function isNoResultsOutput(
  output: FeatureRequestOutput,
): output is NoResultsResponse {
  return (
    output.type === "no_results" ||
    ("suggestions" in output && !("error" in output))
  );
}

export function isErrorOutput(
  output: FeatureRequestOutput,
): output is ErrorResponse {
  return output.type === "error" || "error" in output;
}

/* ------------------------------------------------------------------ */
/*  Accordion metadata                                                 */
/* ------------------------------------------------------------------ */

export function getAccordionTitle(
  toolType: FeatureRequestToolType,
  output: FeatureRequestOutput,
): string {
  if (toolType === "tool-search_feature_requests") {
    if (isSearchResultsOutput(output)) return "Feature requests";
    if (isNoResultsOutput(output)) return "No feature requests found";
    return "Feature request search error";
  }
  if (isCreatedOutput(output)) {
    return output.is_new_issue
      ? "Feature request created"
      : "Added to feature request";
  }
  if (isErrorOutput(output)) return "Feature request error";
  return "Feature request";
}

/* ------------------------------------------------------------------ */
/*  Animation text                                                     */
/* ------------------------------------------------------------------ */

interface AnimationPart {
  type: FeatureRequestToolType;
  state: ToolUIPart["state"];
  input?: unknown;
  output?: unknown;
}

export function getAnimationText(part: AnimationPart): string {
  if (part.type === "tool-search_feature_requests") {
    const query = (part.input as { query?: string } | undefined)?.query?.trim();
    const queryText = query ? ` for "${query}"` : "";

    switch (part.state) {
      case "input-streaming":
      case "input-available":
        return `Searching feature requests${queryText}`;
      case "output-available": {
        const output = parseOutput(part.output);
        if (!output) return `Searching feature requests${queryText}`;
        if (isSearchResultsOutput(output)) {
          return `Found ${output.count} feature request${output.count === 1 ? "" : "s"}${queryText}`;
        }
        if (isNoResultsOutput(output))
          return `No feature requests found${queryText}`;
        return `Error searching feature requests${queryText}`;
      }
      case "output-error":
        return `Error searching feature requests${queryText}`;
      default:
        return "Searching feature requests";
    }
  }

  // create_feature_request
  const title = (part.input as { title?: string } | undefined)?.title?.trim();
  const titleText = title ? ` "${title}"` : "";

  switch (part.state) {
    case "input-streaming":
    case "input-available":
      return `Creating feature request${titleText}`;
    case "output-available": {
      const output = parseOutput(part.output);
      if (!output) return `Creating feature request${titleText}`;
      if (isCreatedOutput(output)) {
        return output.is_new_issue
          ? "Feature request created"
          : "Added to existing feature request";
      }
      if (isErrorOutput(output)) return "Error creating feature request";
      return `Created feature request${titleText}`;
    }
    case "output-error":
      return "Error creating feature request";
    default:
      return "Creating feature request";
  }
}

/* ------------------------------------------------------------------ */
/*  Icons                                                              */
/* ------------------------------------------------------------------ */

export function ToolIcon({
  toolType,
  isStreaming,
  isError,
}: {
  toolType: FeatureRequestToolType;
  isStreaming?: boolean;
  isError?: boolean;
}) {
  const IconComponent =
    toolType === "tool-create_feature_request"
      ? PlusCircleIcon
      : MagnifyingGlassIcon;

  return (
    <IconComponent
      size={14}
      weight="regular"
      className={
        isError
          ? "text-red-500"
          : isStreaming
            ? "text-neutral-500"
            : "text-neutral-400"
      }
    />
  );
}

export function AccordionIcon({
  toolType,
}: {
  toolType: FeatureRequestToolType;
}) {
  const IconComponent =
    toolType === "tool-create_feature_request"
      ? CheckCircleIcon
      : LightbulbIcon;
  return <IconComponent size={32} weight="light" />;
}
