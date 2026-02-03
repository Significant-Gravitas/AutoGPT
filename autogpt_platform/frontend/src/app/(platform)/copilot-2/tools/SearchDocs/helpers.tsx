import { ToolUIPart } from "ai";
import {
  CheckCircleIcon,
  CircleNotchIcon,
  XCircleIcon,
} from "@phosphor-icons/react";

export interface SearchDocsInput {
  query: string;
}

export interface GetDocPageInput {
  path: string;
}

export interface DocSearchResult {
  title: string;
  path: string;
  section: string;
  snippet: string;
  score: number;
  doc_url?: string | null;
}

export interface DocSearchResultsOutput {
  type: "doc_search_results";
  message: string;
  session_id?: string;
  results: DocSearchResult[];
  count: number;
  query: string;
}

export interface DocPageOutput {
  type: "doc_page";
  message: string;
  session_id?: string;
  title: string;
  path: string;
  content: string;
  doc_url?: string | null;
}

export interface NoResultsOutput {
  type: "no_results";
  message: string;
  suggestions?: string[];
  session_id?: string;
}

export interface ErrorOutput {
  type: "error";
  message: string;
  error?: string;
  session_id?: string;
}

export type DocsToolOutput =
  | DocSearchResultsOutput
  | DocPageOutput
  | NoResultsOutput
  | ErrorOutput;

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
      return JSON.parse(trimmed) as DocsToolOutput;
    } catch {
      return null;
    }
  }
  if (typeof output === "object") {
    return output as DocsToolOutput;
  }
  return null;
}

export function getDocsToolOutput(part: unknown): DocsToolOutput | null {
  if (!part || typeof part !== "object") return null;
  return parseOutput((part as { output?: unknown }).output);
}

export function getDocsToolTitle(
  toolType: DocsToolType,
  output: DocsToolOutput,
): string {
  if (toolType === "tool-search_docs") {
    if (output.type === "doc_search_results") return "Documentation results";
    if (output.type === "no_results") return "No documentation found";
    return "Documentation search error";
  }

  if (output.type === "doc_page") return "Documentation page";
  if (output.type === "no_results") return "No documentation found";
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
      switch (part.state) {
        case "input-streaming":
          return "Searching docs for you";
        case "input-available": {
          const query = (
            part.input as SearchDocsInput | undefined
          )?.query?.trim();
          return query ? `Searching docs for "${query}"` : "Searching docs";
        }
        case "output-available": {
          const output = parseOutput(part.output);
          const query = (
            part.input as SearchDocsInput | undefined
          )?.query?.trim();
          if (!output) return "Found documentation";
          if (output.type === "doc_search_results") {
            const count = output.count ?? output.results.length;
            return query
              ? `Found ${count} doc result${count === 1 ? "" : "s"} for "${query}"`
              : `Found ${count} doc result${count === 1 ? "" : "s"}`;
          }
          if (output.type === "no_results") {
            return query ? `No docs found for "${query}"` : "No docs found";
          }
          return "Error searching docs";
        }
        case "output-error":
          return "Error searching docs";
        default:
          return "Processing";
      }
    }
    case "tool-get_doc_page": {
      switch (part.state) {
        case "input-streaming":
          return "Loading documentation page";
        case "input-available": {
          const path = (
            part.input as GetDocPageInput | undefined
          )?.path?.trim();
          return path ? `Loading "${path}"` : "Loading documentation page";
        }
        case "output-available": {
          const output = parseOutput(part.output);
          if (!output) return "Loaded documentation page";
          if (output.type === "doc_page") return `Loaded "${output.title}"`;
          if (output.type === "no_results")
            return "Documentation page not found";
          return "Error loading documentation page";
        }
        case "output-error":
          return "Error loading documentation page";
        default:
          return "Processing";
      }
    }
  }

  return "Processing";
}

export function StateIcon({ state }: { state: ToolUIPart["state"] }) {
  switch (state) {
    case "input-streaming":
    case "input-available":
      return (
        <CircleNotchIcon
          className="h-4 w-4 animate-spin text-muted-foreground"
          weight="bold"
        />
      );
    case "output-available":
      return <CheckCircleIcon className="h-4 w-4 text-green-500" />;
    case "output-error":
      return <XCircleIcon className="h-4 w-4 text-red-500" />;
    default:
      return null;
  }
}

export function toDocsUrl(path: string): string {
  const urlPath = path.includes(".")
    ? path.slice(0, path.lastIndexOf("."))
    : path;
  return `https://docs.agpt.co/${urlPath}`;
}
