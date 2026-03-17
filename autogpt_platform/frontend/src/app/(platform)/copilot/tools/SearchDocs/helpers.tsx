import type { DocPageResponse } from "@/app/api/__generated__/models/docPageResponse";
import type { DocSearchResultsResponse } from "@/app/api/__generated__/models/docSearchResultsResponse";
import type { ErrorResponse } from "@/app/api/__generated__/models/errorResponse";
import type { NoResultsResponse } from "@/app/api/__generated__/models/noResultsResponse";
import { ResponseType } from "@/app/api/__generated__/models/responseType";
import {
  ArticleIcon,
  FileMagnifyingGlassIcon,
  FileTextIcon,
} from "@phosphor-icons/react";
import { ToolUIPart } from "ai";

export interface SearchDocsInput {
  query: string;
}

export interface GetDocPageInput {
  path: string;
}

export type DocsToolOutput =
  | DocSearchResultsResponse
  | DocPageResponse
  | NoResultsResponse
  | ErrorResponse;

export type DocsToolType = "tool-search_docs" | "tool-get_doc_page" | string;

export function getToolLabel(toolType: DocsToolType): string {
  switch (toolType) {
    case "tool-search_docs":
      return "Docs";
    case "tool-get_doc_page":
      return "Docs page";
    default:
      return "Docs";
  }
}

function parseOutput(output: unknown): DocsToolOutput | null {
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
      type === ResponseType.doc_search_results ||
      type === ResponseType.doc_page ||
      type === ResponseType.no_results ||
      type === ResponseType.error
    ) {
      return output as DocsToolOutput;
    }
    if ("results" in output && "query" in output)
      return output as DocSearchResultsResponse;
    if ("content" in output && "path" in output)
      return output as DocPageResponse;
    if ("suggestions" in output && !("error" in output))
      return output as NoResultsResponse;
    if ("error" in output || "details" in output)
      return output as ErrorResponse;
  }
  return null;
}

export function getDocsToolOutput(part: unknown): DocsToolOutput | null {
  if (!part || typeof part !== "object") return null;
  return parseOutput((part as { output?: unknown }).output);
}

export function isDocSearchResultsOutput(
  output: DocsToolOutput,
): output is DocSearchResultsResponse {
  return output.type === ResponseType.doc_search_results || "results" in output;
}

export function isDocPageOutput(
  output: DocsToolOutput,
): output is DocPageResponse {
  return output.type === ResponseType.doc_page || "content" in output;
}

export function isNoResultsOutput(
  output: DocsToolOutput,
): output is NoResultsResponse {
  return (
    output.type === ResponseType.no_results ||
    ("suggestions" in output && !("error" in output))
  );
}

export function isErrorOutput(output: DocsToolOutput): output is ErrorResponse {
  return output.type === ResponseType.error || "error" in output;
}

export function getDocsToolTitle(
  toolType: DocsToolType,
  output: DocsToolOutput,
): string {
  if (toolType === "tool-search_docs") {
    if (isDocSearchResultsOutput(output)) return "Documentation results";
    if (isNoResultsOutput(output)) return "No documentation found";
    return "Documentation search error";
  }

  if (isDocPageOutput(output)) return "Documentation page";
  if (isNoResultsOutput(output)) return "No documentation found";
  return "Documentation page error";
}

export function getAnimationText(part: {
  type: DocsToolType;
  state: ToolUIPart["state"];
  input?: unknown;
  output?: unknown;
}): string {
  switch (part.type) {
    case "tool-search_docs": {
      const query = (part.input as SearchDocsInput | undefined)?.query?.trim();
      const queryText = query ? ` for "${query}"` : "";

      switch (part.state) {
        case "input-streaming":
        case "input-available":
          return `Searching documentation${queryText}`;
        case "output-available": {
          const output = parseOutput(part.output);
          if (!output) return `Searching documentation${queryText}`;
          if (isDocSearchResultsOutput(output)) {
            const count = output.count ?? output.results.length;
            return `Found ${count} result${count === 1 ? "" : "s"}${queryText}`;
          }
          if (isNoResultsOutput(output)) {
            return `No results found${queryText}`;
          }
          return `Error searching documentation${queryText}`;
        }
        case "output-error":
          return `Error searching documentation${queryText}`;
        default:
          return "Searching documentation";
      }
    }
    case "tool-get_doc_page": {
      const path = (part.input as GetDocPageInput | undefined)?.path?.trim();
      const pathText = path ? ` "${path}"` : "";

      switch (part.state) {
        case "input-streaming":
        case "input-available":
          return `Loading documentation page${pathText}`;
        case "output-available": {
          const output = parseOutput(part.output);
          if (!output) return `Loading documentation page${pathText}`;
          if (isDocPageOutput(output)) return `Loaded "${output.title}"`;
          if (isNoResultsOutput(output)) return "Documentation page not found";
          return "Error loading documentation page";
        }
        case "output-error":
          return "Error loading documentation page";
        default:
          return "Loading documentation page";
      }
    }
  }

  return "Processing";
}

export function ToolIcon({
  toolType,
  isStreaming,
  isError,
}: {
  toolType: DocsToolType;
  isStreaming?: boolean;
  isError?: boolean;
}) {
  const IconComponent =
    toolType === "tool-get_doc_page" ? FileTextIcon : FileMagnifyingGlassIcon;

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

export function AccordionIcon({ toolType }: { toolType: DocsToolType }) {
  const IconComponent =
    toolType === "tool-get_doc_page" ? ArticleIcon : FileTextIcon;
  return <IconComponent size={32} weight="light" />;
}

export function toDocsUrl(path: string): string {
  const urlPath = path.includes(".")
    ? path.slice(0, path.lastIndexOf("."))
    : path;
  return `https://docs.agpt.co/${urlPath}`;
}
