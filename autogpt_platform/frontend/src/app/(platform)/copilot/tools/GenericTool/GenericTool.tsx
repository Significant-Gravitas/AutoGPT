"use client";

import React from "react";
import { ToolUIPart } from "ai";
import {
  CheckCircleIcon,
  CircleDashedIcon,
  CircleIcon,
  FileIcon,
  FilesIcon,
  GearIcon,
  GlobeIcon,
  ListChecksIcon,
  MagnifyingGlassIcon,
  PencilSimpleIcon,
  TerminalIcon,
  TrashIcon,
  WarningDiamondIcon,
} from "@phosphor-icons/react";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
import {
  ContentCodeBlock,
  ContentMessage,
} from "../../components/ToolAccordion/AccordionContent";
import { OrbitLoader } from "../../components/OrbitLoader/OrbitLoader";

interface Props {
  part: ToolUIPart;
}

/* ------------------------------------------------------------------ */
/*  Tool name helpers                                                  */
/* ------------------------------------------------------------------ */

function extractToolName(part: ToolUIPart): string {
  return part.type.replace(/^tool-/, "");
}

function formatToolName(name: string): string {
  return name.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

/* ------------------------------------------------------------------ */
/*  Tool categorization                                                */
/* ------------------------------------------------------------------ */

type ToolCategory =
  | "bash"
  | "web"
  | "file-read"
  | "file-write"
  | "file-delete"
  | "file-list"
  | "search"
  | "edit"
  | "todo"
  | "other";

function getToolCategory(toolName: string): ToolCategory {
  switch (toolName) {
    case "bash_exec":
      return "bash";
    case "web_fetch":
    case "WebSearch":
    case "WebFetch":
      return "web";
    case "read_workspace_file":
    case "Read":
      return "file-read";
    case "write_workspace_file":
    case "Write":
      return "file-write";
    case "delete_workspace_file":
      return "file-delete";
    case "list_workspace_files":
    case "Glob":
      return "file-list";
    case "Grep":
      return "search";
    case "Edit":
      return "edit";
    case "TodoWrite":
      return "todo";
    default:
      return "other";
  }
}

/* ------------------------------------------------------------------ */
/*  Tool icon                                                          */
/* ------------------------------------------------------------------ */

function ToolIcon({
  category,
  isStreaming,
  isError,
}: {
  category: ToolCategory;
  isStreaming: boolean;
  isError: boolean;
}) {
  if (isError) {
    return (
      <WarningDiamondIcon size={14} weight="regular" className="text-red-500" />
    );
  }
  if (isStreaming) {
    return <OrbitLoader size={14} />;
  }

  const iconClass = "text-neutral-400";
  switch (category) {
    case "bash":
      return <TerminalIcon size={14} weight="regular" className={iconClass} />;
    case "web":
      return <GlobeIcon size={14} weight="regular" className={iconClass} />;
    case "file-read":
      return <FileIcon size={14} weight="regular" className={iconClass} />;
    case "file-write":
      return <FileIcon size={14} weight="regular" className={iconClass} />;
    case "file-delete":
      return <TrashIcon size={14} weight="regular" className={iconClass} />;
    case "file-list":
      return <FilesIcon size={14} weight="regular" className={iconClass} />;
    case "search":
      return (
        <MagnifyingGlassIcon size={14} weight="regular" className={iconClass} />
      );
    case "edit":
      return (
        <PencilSimpleIcon size={14} weight="regular" className={iconClass} />
      );
    case "todo":
      return (
        <ListChecksIcon size={14} weight="regular" className={iconClass} />
      );
    default:
      return <GearIcon size={14} weight="regular" className={iconClass} />;
  }
}

/* ------------------------------------------------------------------ */
/*  Accordion icon (larger, for the accordion header)                  */
/* ------------------------------------------------------------------ */

function AccordionIcon({ category }: { category: ToolCategory }) {
  switch (category) {
    case "bash":
      return <TerminalIcon size={32} weight="light" />;
    case "web":
      return <GlobeIcon size={32} weight="light" />;
    case "file-read":
    case "file-write":
      return <FileIcon size={32} weight="light" />;
    case "file-delete":
      return <TrashIcon size={32} weight="light" />;
    case "file-list":
      return <FilesIcon size={32} weight="light" />;
    case "search":
      return <MagnifyingGlassIcon size={32} weight="light" />;
    case "edit":
      return <PencilSimpleIcon size={32} weight="light" />;
    case "todo":
      return <ListChecksIcon size={32} weight="light" />;
    default:
      return <GearIcon size={32} weight="light" />;
  }
}

/* ------------------------------------------------------------------ */
/*  Input extraction                                                   */
/* ------------------------------------------------------------------ */

function getInputSummary(toolName: string, input: unknown): string | null {
  if (!input || typeof input !== "object") return null;
  const inp = input as Record<string, unknown>;

  switch (toolName) {
    case "bash_exec":
      return typeof inp.command === "string" ? inp.command : null;
    case "web_fetch":
    case "WebFetch":
      return typeof inp.url === "string" ? inp.url : null;
    case "WebSearch":
      return typeof inp.query === "string" ? inp.query : null;
    case "read_workspace_file":
    case "Read":
      return (
        (typeof inp.file_path === "string" ? inp.file_path : null) ??
        (typeof inp.path === "string" ? inp.path : null)
      );
    case "write_workspace_file":
    case "Write":
      return (
        (typeof inp.file_path === "string" ? inp.file_path : null) ??
        (typeof inp.path === "string" ? inp.path : null)
      );
    case "delete_workspace_file":
      return typeof inp.file_path === "string" ? inp.file_path : null;
    case "Glob":
      return typeof inp.pattern === "string" ? inp.pattern : null;
    case "Grep":
      return typeof inp.pattern === "string" ? inp.pattern : null;
    case "Edit":
      return typeof inp.file_path === "string" ? inp.file_path : null;
    case "TodoWrite": {
      // Extract the in-progress task name for the status line
      const todos = Array.isArray(inp.todos) ? inp.todos : [];
      const active = todos.find(
        (t: Record<string, unknown>) => t.status === "in_progress",
      );
      if (active && typeof active.activeForm === "string")
        return active.activeForm;
      if (active && typeof active.content === "string") return active.content;
      return null;
    }
    default:
      return null;
  }
}

function truncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen).trimEnd() + "…";
}

/* ------------------------------------------------------------------ */
/*  Animation text                                                     */
/* ------------------------------------------------------------------ */

function getAnimationText(part: ToolUIPart, category: ToolCategory): string {
  const toolName = extractToolName(part);
  const summary = getInputSummary(toolName, part.input);
  const shortSummary = summary ? truncate(summary, 60) : null;

  switch (part.state) {
    case "input-streaming":
    case "input-available": {
      switch (category) {
        case "bash":
          return shortSummary ? `Running: ${shortSummary}` : "Running command…";
        case "web":
          if (toolName === "WebSearch") {
            return shortSummary
              ? `Searching "${shortSummary}"`
              : "Searching the web…";
          }
          return shortSummary
            ? `Fetching ${shortSummary}`
            : "Fetching web content…";
        case "file-read":
          return shortSummary ? `Reading ${shortSummary}` : "Reading file…";
        case "file-write":
          return shortSummary ? `Writing ${shortSummary}` : "Writing file…";
        case "file-delete":
          return shortSummary ? `Deleting ${shortSummary}` : "Deleting file…";
        case "file-list":
          return shortSummary ? `Listing ${shortSummary}` : "Listing files…";
        case "search":
          return shortSummary
            ? `Searching for "${shortSummary}"`
            : "Searching…";
        case "edit":
          return shortSummary ? `Editing ${shortSummary}` : "Editing file…";
        case "todo":
          return shortSummary ? `${shortSummary}` : "Updating task list…";
        default:
          return `Running ${formatToolName(toolName)}…`;
      }
    }
    case "output-available": {
      switch (category) {
        case "bash": {
          const exitCode = getExitCode(part.output);
          if (exitCode !== null && exitCode !== 0) {
            return `Command exited with code ${exitCode}`;
          }
          return shortSummary ? `Ran: ${shortSummary}` : "Command completed";
        }
        case "web":
          if (toolName === "WebSearch") {
            return shortSummary
              ? `Searched "${shortSummary}"`
              : "Web search completed";
          }
          return shortSummary
            ? `Fetched ${shortSummary}`
            : "Fetched web content";
        case "file-read":
          return shortSummary ? `Read ${shortSummary}` : "File read completed";
        case "file-write":
          return shortSummary ? `Wrote ${shortSummary}` : "File written";
        case "file-delete":
          return shortSummary ? `Deleted ${shortSummary}` : "File deleted";
        case "file-list":
          return "Listed files";
        case "search":
          return shortSummary
            ? `Searched for "${shortSummary}"`
            : "Search completed";
        case "edit":
          return shortSummary ? `Edited ${shortSummary}` : "Edit completed";
        case "todo":
          return "Updated task list";
        default:
          return `${formatToolName(toolName)} completed`;
      }
    }
    case "output-error": {
      switch (category) {
        case "bash":
          return "Command failed";
        case "web":
          return toolName === "WebSearch" ? "Search failed" : "Fetch failed";
        default:
          return `${formatToolName(toolName)} failed`;
      }
    }
    default:
      return `Running ${formatToolName(toolName)}…`;
  }
}

/* ------------------------------------------------------------------ */
/*  Output parsing helpers                                             */
/* ------------------------------------------------------------------ */

function parseOutput(output: unknown): Record<string, unknown> | null {
  if (!output) return null;
  if (typeof output === "object") return output as Record<string, unknown>;
  if (typeof output === "string") {
    const trimmed = output.trim();
    if (!trimmed) return null;
    try {
      const parsed = JSON.parse(trimmed);
      if (
        typeof parsed === "object" &&
        parsed !== null &&
        !Array.isArray(parsed)
      )
        return parsed;
    } catch {
      // Return as a message wrapper for plain text output
      return { _raw: trimmed };
    }
  }
  return null;
}

/**
 * Extract text from MCP-style content blocks.
 * SDK built-in tools (WebSearch, etc.) may return `{content: [{type:"text", text:"..."}]}`.
 */
function extractMcpText(output: Record<string, unknown>): string | null {
  if (Array.isArray(output.content)) {
    const texts = (output.content as Array<Record<string, unknown>>)
      .filter((b) => b.type === "text" && typeof b.text === "string")
      .map((b) => b.text as string);
    if (texts.length > 0) return texts.join("\n");
  }
  return null;
}

function getExitCode(output: unknown): number | null {
  const parsed = parseOutput(output);
  if (!parsed) return null;
  if (typeof parsed.exit_code === "number") return parsed.exit_code;
  return null;
}

function getStringField(
  obj: Record<string, unknown>,
  ...keys: string[]
): string | null {
  for (const key of keys) {
    if (typeof obj[key] === "string" && obj[key].length > 0)
      return obj[key] as string;
  }
  return null;
}

/* ------------------------------------------------------------------ */
/*  Accordion content per tool category                                */
/* ------------------------------------------------------------------ */

interface AccordionData {
  title: string;
  description?: string;
  content: React.ReactNode;
}

function getBashAccordionData(
  input: unknown,
  output: Record<string, unknown>,
): AccordionData {
  const inp = (input && typeof input === "object" ? input : {}) as Record<
    string,
    unknown
  >;
  const command = typeof inp.command === "string" ? inp.command : "Command";

  const stdout = getStringField(output, "stdout");
  const stderr = getStringField(output, "stderr");
  const exitCode =
    typeof output.exit_code === "number" ? output.exit_code : null;
  const timedOut = output.timed_out === true;
  const message = getStringField(output, "message");

  const title = timedOut
    ? "Command timed out"
    : exitCode !== null && exitCode !== 0
      ? `Command failed (exit ${exitCode})`
      : "Command output";

  return {
    title,
    description: truncate(command, 80),
    content: (
      <div className="space-y-2">
        {stdout && (
          <div>
            <p className="mb-1 text-xs font-medium text-slate-500">stdout</p>
            <ContentCodeBlock>{truncate(stdout, 2000)}</ContentCodeBlock>
          </div>
        )}
        {stderr && (
          <div>
            <p className="mb-1 text-xs font-medium text-slate-500">stderr</p>
            <ContentCodeBlock>{truncate(stderr, 1000)}</ContentCodeBlock>
          </div>
        )}
        {!stdout && !stderr && message && (
          <ContentMessage>{message}</ContentMessage>
        )}
      </div>
    ),
  };
}

function getWebAccordionData(
  input: unknown,
  output: Record<string, unknown>,
): AccordionData {
  const inp = (input && typeof input === "object" ? input : {}) as Record<
    string,
    unknown
  >;
  const url =
    getStringField(inp as Record<string, unknown>, "url", "query") ??
    "Web content";

  // Try direct string fields first, then MCP content blocks, then raw JSON
  let content = getStringField(output, "content", "text", "_raw");
  if (!content) content = extractMcpText(output);
  if (!content) {
    // Fallback: render the raw JSON so the accordion isn't empty
    try {
      const raw = JSON.stringify(output, null, 2);
      if (raw !== "{}") content = raw;
    } catch {
      /* empty */
    }
  }

  const statusCode =
    typeof output.status_code === "number" ? output.status_code : null;
  const message = getStringField(output, "message");

  return {
    title: statusCode
      ? `Response (${statusCode})`
      : url
        ? "Web fetch"
        : "Search results",
    description: truncate(url, 80),
    content: content ? (
      <ContentCodeBlock>{truncate(content, 2000)}</ContentCodeBlock>
    ) : message ? (
      <ContentMessage>{message}</ContentMessage>
    ) : Object.keys(output).length > 0 ? (
      <ContentCodeBlock>
        {truncate(JSON.stringify(output, null, 2), 2000)}
      </ContentCodeBlock>
    ) : null,
  };
}

function getFileAccordionData(
  input: unknown,
  output: Record<string, unknown>,
): AccordionData {
  const inp = (input && typeof input === "object" ? input : {}) as Record<
    string,
    unknown
  >;
  const filePath =
    getStringField(
      inp as Record<string, unknown>,
      "file_path",
      "path",
      "pattern",
    ) ?? "File";
  const content = getStringField(
    output,
    "content",
    "text",
    "preview",
    "content_preview",
    "_raw",
  );
  const message = getStringField(output, "message");

  // Handle base64 content from workspace files
  let displayContent = content;
  if (output.content_base64 && typeof output.content_base64 === "string") {
    try {
      const bytes = Uint8Array.from(atob(output.content_base64), (c) =>
        c.charCodeAt(0),
      );
      displayContent = new TextDecoder().decode(bytes);
    } catch {
      displayContent = "[Binary content]";
    }
  }

  // Handle MCP-style content blocks from SDK tools (Read, Glob, Grep, Edit)
  if (!displayContent) {
    displayContent = extractMcpText(output);
  }

  // For Glob/list results, try to show file list
  // Files can be either strings (from Glob) or objects (from list_workspace_files)
  const files = Array.isArray(output.files) ? output.files : null;

  // Format file list for display
  let fileListText: string | null = null;
  if (files && files.length > 0) {
    const fileLines = files.map((f: unknown) => {
      if (typeof f === "string") {
        return f;
      }
      if (typeof f === "object" && f !== null) {
        const fileObj = f as Record<string, unknown>;
        // Workspace file format: path (size, mime_type)
        const filePath =
          typeof fileObj.path === "string"
            ? fileObj.path
            : typeof fileObj.name === "string"
              ? fileObj.name
              : "unknown";
        const mimeType =
          typeof fileObj.mime_type === "string" ? fileObj.mime_type : "unknown";
        const size =
          typeof fileObj.size_bytes === "number"
            ? ` (${(fileObj.size_bytes / 1024).toFixed(1)} KB, ${mimeType})`
            : "";
        return `${filePath}${size}`;
      }
      return String(f);
    });
    fileListText = fileLines.join("\n");
  }

  return {
    title: message ?? "File output",
    description: truncate(filePath, 80),
    content: (
      <div className="space-y-2">
        {displayContent && (
          <ContentCodeBlock>{truncate(displayContent, 2000)}</ContentCodeBlock>
        )}
        {fileListText && (
          <ContentCodeBlock>{truncate(fileListText, 2000)}</ContentCodeBlock>
        )}
        {!displayContent && !fileListText && message && (
          <ContentMessage>{message}</ContentMessage>
        )}
      </div>
    ),
  };
}

interface TodoItem {
  content: string;
  status: "pending" | "in_progress" | "completed";
  activeForm?: string;
}

function getTodoAccordionData(input: unknown): AccordionData {
  const inp = (input && typeof input === "object" ? input : {}) as Record<
    string,
    unknown
  >;
  const todos: TodoItem[] = Array.isArray(inp.todos)
    ? inp.todos.filter(
        (t: unknown): t is TodoItem =>
          typeof t === "object" &&
          t !== null &&
          typeof (t as TodoItem).content === "string",
      )
    : [];

  const completed = todos.filter((t) => t.status === "completed").length;
  const total = todos.length;

  return {
    title: "Task list",
    description: `${completed}/${total} completed`,
    content: (
      <div className="space-y-1 py-1">
        {todos.map((todo, i) => (
          <div key={i} className="flex items-start gap-2 text-xs">
            <span className="mt-0.5 flex-shrink-0">
              {todo.status === "completed" ? (
                <CheckCircleIcon
                  size={14}
                  weight="fill"
                  className="text-green-500"
                />
              ) : todo.status === "in_progress" ? (
                <CircleDashedIcon
                  size={14}
                  weight="bold"
                  className="text-blue-500"
                />
              ) : (
                <CircleIcon
                  size={14}
                  weight="regular"
                  className="text-neutral-400"
                />
              )}
            </span>
            <span
              className={
                todo.status === "completed"
                  ? "text-muted-foreground line-through"
                  : todo.status === "in_progress"
                    ? "font-medium text-foreground"
                    : "text-muted-foreground"
              }
            >
              {todo.content}
            </span>
          </div>
        ))}
      </div>
    ),
  };
}

function getDefaultAccordionData(
  output: Record<string, unknown>,
): AccordionData {
  const message = getStringField(output, "message");
  const raw = output._raw;
  const mcpText = extractMcpText(output);

  let displayContent: string;
  if (typeof raw === "string") {
    displayContent = raw;
  } else if (mcpText) {
    displayContent = mcpText;
  } else if (message) {
    displayContent = message;
  } else {
    try {
      displayContent = JSON.stringify(output, null, 2);
    } catch {
      displayContent = String(output);
    }
  }

  return {
    title: "Output",
    description: message ?? undefined,
    content: (
      <ContentCodeBlock>{truncate(displayContent, 2000)}</ContentCodeBlock>
    ),
  };
}

function getAccordionData(
  category: ToolCategory,
  input: unknown,
  output: Record<string, unknown>,
): AccordionData {
  switch (category) {
    case "bash":
      return getBashAccordionData(input, output);
    case "web":
      return getWebAccordionData(input, output);
    case "file-read":
    case "file-write":
    case "file-delete":
    case "file-list":
    case "search":
    case "edit":
      return getFileAccordionData(input, output);
    case "todo":
      return getTodoAccordionData(input);
    default:
      return getDefaultAccordionData(output);
  }
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export function GenericTool({ part }: Props) {
  const toolName = extractToolName(part);
  const category = getToolCategory(toolName);
  const isStreaming =
    part.state === "input-streaming" || part.state === "input-available";
  const isError = part.state === "output-error";
  const text = getAnimationText(part, category);

  const output = parseOutput(part.output);
  const hasOutput =
    part.state === "output-available" &&
    !!output &&
    Object.keys(output).length > 0;
  const hasError = isError && !!output;

  // TodoWrite: always show accordion from input (the todo list lives in input)
  const hasTodoInput =
    category === "todo" &&
    part.input &&
    typeof part.input === "object" &&
    Array.isArray((part.input as Record<string, unknown>).todos);

  const showAccordion = hasOutput || hasError || hasTodoInput;
  const accordionData = showAccordion
    ? getAccordionData(category, part.input, output ?? {})
    : null;

  return (
    <div className="py-2">
      {/* Only show loading text when NOT showing accordion */}
      {!showAccordion && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <ToolIcon
            category={category}
            isStreaming={isStreaming}
            isError={isError}
          />
          <MorphingTextAnimation
            text={text}
            className={isError ? "text-red-500" : undefined}
          />
        </div>
      )}

      {showAccordion && accordionData ? (
        <ToolAccordion
          icon={<AccordionIcon category={category} />}
          title={accordionData.title}
          description={accordionData.description}
          titleClassName={isError ? "text-red-500" : undefined}
          defaultExpanded={category === "todo"}
        >
          {accordionData.content}
        </ToolAccordion>
      ) : null}
    </div>
  );
}
