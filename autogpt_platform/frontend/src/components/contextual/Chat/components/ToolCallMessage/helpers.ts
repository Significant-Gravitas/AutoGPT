import type { ToolArguments, ToolResult } from "@/types/chat";
import {
  BrainIcon,
  EyeIcon,
  FileMagnifyingGlassIcon,
  FileTextIcon,
  MagnifyingGlassIcon,
  PackageIcon,
  PencilLineIcon,
  PlayIcon,
  PlusIcon,
  SquaresFourIcon,
  type Icon,
} from "@phosphor-icons/react";

/**
 * Maps internal tool names to human-friendly action phrases (present continuous).
 * Used for tool call messages to indicate what action is currently happening.
 *
 * @param toolName - The internal tool name from the backend
 * @returns A human-friendly action phrase in present continuous tense
 */
export function getToolActionPhrase(toolName: string): string {
  const normalizedName = toolName.trim();
  if (!normalizedName) return "Executing";
  if (normalizedName.toLowerCase().startsWith("executing")) {
    return normalizedName;
  }
  if (normalizedName.toLowerCase() === "unknown") return "Executing";
  const toolActionPhrases: Record<string, string> = {
    add_understanding: "Updating your business information",
    create_agent: "Creating a new agent",
    edit_agent: "Editing the agent",
    find_agent: "Looking for agents in the marketplace",
    find_block: "Searching for blocks",
    find_library_agent: "Looking for library agents",
    run_agent: "Running the agent",
    run_block: "Running the block",
    view_agent_output: "Retrieving agent output",
    search_docs: "Searching documentation",
    get_doc_page: "Loading documentation page",
    agent_carousel: "Looking for agents in the marketplace",
    execution_started: "Running the agent",
    get_required_setup_info: "Getting setup requirements",
    schedule_agent: "Scheduling the agent to run",
  };

  // Return mapped phrase or generate human-friendly fallback
  return (
    toolActionPhrases[toolName] ||
    toolName
      .replace(/_/g, " ")
      .replace(/\b\w/g, (l) => l.toUpperCase())
      .replace(/^/, "Executing ")
  );
}

/** Parse tool result to extract structured data */
function parseToolResult(result: ToolResult): Record<string, unknown> | null {
  if (!result) return null;
  if (typeof result === "string") {
    try {
      return JSON.parse(result);
    } catch {
      return null;
    }
  }
  if (typeof result === "object") {
    return result as Record<string, unknown>;
  }
  return null;
}

/**
 * Formats tool call arguments into user-friendly text.
 * Handles different tool types and formats their arguments nicely.
 *
 * @param toolName - The tool name
 * @param args - The tool arguments
 * @param toolResponse - Optional tool response to extract additional info (e.g., block_name)
 * @returns Formatted user-friendly text to append to action phrase
 */
export function formatToolArguments(
  toolName: string,
  args: ToolArguments | undefined,
  toolResponse?: { result: ToolResult } | null,
): string {
  if (!args || Object.keys(args).length === 0) {
    return "";
  }

  switch (toolName) {
    case "find_agent":
    case "find_library_agent":
    case "agent_carousel":
      if (args.query) {
        return ` matching "${args.query as string}"`;
      }
      break;

    case "find_block":
      if (args.query) {
        return ` matching "${args.query as string}"`;
      }
      break;

    case "search_docs":
      if (args.query) {
        return ` for "${args.query as string}"`;
      }
      break;

    case "get_doc_page":
      if (args.path) {
        return ` "${args.path as string}"`;
      }
      break;

    case "run_agent":
      if (args.username_agent_slug) {
        return ` "${args.username_agent_slug as string}"`;
      }
      if (args.library_agent_id) {
        return ` (library agent)`;
      }
      break;

    case "run_block": {
      // Prefer block_name from tool response (BlockOutputResponse includes it)
      const blockResult = toolResponse
        ? parseToolResult(toolResponse.result)
        : null;
      if (
        blockResult?.block_name &&
        typeof blockResult.block_name === "string"
      ) {
        return ` "${blockResult.block_name}"`;
      }
      // Fallback to args if response not available
      if (args.block_name) {
        return ` "${args.block_name as string}"`;
      }
      if (args.name) {
        return ` "${args.name as string}"`;
      }
      if (args.block_id) {
        return ` "${args.block_id as string}"`;
      }
      break;
    }

    case "view_agent_output":
      if (args.library_agent_id) {
        return ` (library agent)`;
      }
      if (args.username_agent_slug) {
        return ` "${args.username_agent_slug as string}"`;
      }
      break;

    case "create_agent":
    case "edit_agent":
      if (args.name) {
        return ` "${args.name as string}"`;
      }
      break;

    case "add_understanding":
      const understandingFields = Object.entries(args)
        .filter(
          ([_, value]) => value !== null && value !== undefined && value !== "",
        )
        .map(([key, value]) => {
          if (key === "user_name" && typeof value === "string") {
            return `for ${value}`;
          }
          if (typeof value === "string") {
            return `${key}: ${value}`;
          }
          if (Array.isArray(value) && value.length > 0) {
            return `${key}: ${value.slice(0, 2).join(", ")}${value.length > 2 ? ` (+${value.length - 2} more)` : ""}`;
          }
          return key;
        });
      if (understandingFields.length > 0) {
        return ` ${understandingFields[0]}`;
      }
      break;
  }

  return "";
}

/**
 * Maps tool names to their corresponding Phosphor icon components.
 *
 * @param toolName - The tool name from the backend
 * @returns The Icon component for the tool
 */
export function getToolIcon(toolName: string): Icon {
  const iconMap: Record<string, Icon> = {
    add_understanding: BrainIcon,
    create_agent: PlusIcon,
    edit_agent: PencilLineIcon,
    find_agent: SquaresFourIcon,
    find_library_agent: MagnifyingGlassIcon,
    find_block: PackageIcon,
    run_agent: PlayIcon,
    run_block: PlayIcon,
    view_agent_output: EyeIcon,
    search_docs: FileMagnifyingGlassIcon,
    get_doc_page: FileTextIcon,
    agent_carousel: MagnifyingGlassIcon,
    execution_started: PlayIcon,
    get_required_setup_info: SquaresFourIcon,
    schedule_agent: PlayIcon,
  };

  return iconMap[toolName] || SquaresFourIcon;
}

/** Extract the message field from tool result, or format the whole result */
export function getToolResultMessage(result: ToolResult) {
  const parsed = parseToolResult(result);
  if (parsed) {
    // For agent_output, return the output field
    if (parsed.type === "agent_output" && "output" in parsed) {
      return String(parsed.output);
    }
    // Return the message field if it exists
    if (typeof parsed.message === "string") {
      return parsed.message;
    }
  }
  // Fallback to string representation
  if (typeof result === "string") {
    return result;
  }
  return String(result ?? "");
}

/** Format tool result as JSON for debug view */
export function formatToolResultAsJson(result: ToolResult) {
  const parsed = parseToolResult(result);
  if (parsed) {
    return JSON.stringify(parsed, null, 2);
  }
  if (typeof result === "string") {
    return result;
  }
  return String(result ?? "");
}
