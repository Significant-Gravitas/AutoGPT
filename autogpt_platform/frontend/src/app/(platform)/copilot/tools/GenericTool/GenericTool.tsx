"use client";

import React from "react";
import { ToolUIPart } from "ai";
import {
  ArrowsClockwiseIcon,
  CheckCircleIcon,
  CircleDashedIcon,
  CircleIcon,
  FileIcon,
  FilesIcon,
  GearIcon,
  GlobeIcon,
  ListChecksIcon,
  MagnifyingGlassIcon,
  MonitorIcon,
  PencilSimpleIcon,
  RobotIcon,
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
import {
  globalRegistry,
  OutputItem,
} from "@/components/contextual/OutputRenderers";
import type { OutputMetadata } from "@/components/contextual/OutputRenderers";
import {
  TOOL_TASK_OUTPUT,
  type ToolCategory,
  extractToolName,
  getAnimationText,
  getToolCategory,
  truncate,
} from "./helpers";

interface Props {
  part: ToolUIPart;
}

function RenderMedia({
  value,
  metadata,
}: {
  value: string;
  metadata: OutputMetadata;
}) {
  const renderer = globalRegistry.getRenderer(value, metadata);
  if (!renderer) return null;
  return <OutputItem value={value} metadata={metadata} renderer={renderer} />;
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

  const iconClass = "text-green-500";
  switch (category) {
    case "bash":
      return <TerminalIcon size={14} weight="regular" className={iconClass} />;
    case "web":
      return <GlobeIcon size={14} weight="regular" className={iconClass} />;
    case "browser":
      return <MonitorIcon size={14} weight="regular" className={iconClass} />;
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
    case "compaction":
      return (
        <ArrowsClockwiseIcon size={14} weight="regular" className={iconClass} />
      );
    case "agent":
      return <RobotIcon size={14} weight="regular" className={iconClass} />;
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
    case "browser":
      return <MonitorIcon size={32} weight="light" />;
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
    case "compaction":
      return <ArrowsClockwiseIcon size={32} weight="light" />;
    case "agent":
      return <RobotIcon size={32} weight="light" />;
    default:
      return <GearIcon size={32} weight="light" />;
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

  // The command itself is already in the subtitle row above; surface the
  // outcome here so scanning the closed accordion tells the reader "how it
  // ended" at a glance.  Prefer the backend's own first line of output
  // (stderr for failures/timeouts — that's where bash_exec writes
  // "Timed out after Xs" and where shells emit "command not found" etc.,
  // stdout for success) over a terse "exit N" so the reader actually sees
  // WHY the command ended.
  const firstNonEmptyLine = (s: string | null): string | null => {
    if (!s) return null;
    const line = s.split("\n").find((l) => l.trim().length > 0);
    return line ? truncate(line.trim(), 80) : null;
  };
  const stderrPreview = firstNonEmptyLine(stderr);
  const stdoutPreview = firstNonEmptyLine(stdout);
  let description: string | undefined;
  if (timedOut) {
    description = stderrPreview ?? "timed out";
  } else if (exitCode !== null && exitCode !== 0) {
    description = stderrPreview
      ? `status code ${exitCode} · ${stderrPreview}`
      : `status code ${exitCode}`;
  } else if (exitCode === 0) {
    description = stdoutPreview ?? "completed";
  } else {
    // Historical sessions persisted before exit_code/timed_out were added
    // fall through here — fall back to the command preview so the closed
    // accordion still tells the reader what ran.
    description = truncate(command, 80);
  }

  return {
    title,
    description,
    content: (
      <div className="space-y-2">
        {command && (
          <div>
            <p className="mb-1 text-xs font-medium text-slate-500">command</p>
            <ContentCodeBlock>{command}</ContentCodeBlock>
          </div>
        )}
        {stdout && (
          <div>
            <p className="mb-1 text-xs font-medium text-slate-500">stdout</p>
            <ContentCodeBlock>{stdout}</ContentCodeBlock>
          </div>
        )}
        {stderr && (
          <div>
            <p className="mb-1 text-xs font-medium text-slate-500">stderr</p>
            <ContentCodeBlock>{stderr}</ContentCodeBlock>
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
  const query = getStringField(inp, "query");
  const url = getStringField(inp, "url") ?? query ?? "Web content";

  const results = Array.isArray(output.results)
    ? (output.results as Array<Record<string, unknown>>)
    : null;

  if (results) {
    return {
      title: `${results.length} search result${results.length === 1 ? "" : "s"}`,
      description: query ? truncate(query, 80) : undefined,
      content: (
        <div className="space-y-3">
          {results.map((r, i) => {
            const title = getStringField(r, "title") ?? "(untitled)";
            const href = getStringField(r, "url") ?? "";
            const snippet = getStringField(r, "snippet");
            const pageAge = getStringField(r, "page_age");
            return (
              <div key={i} className="text-sm">
                {href ? (
                  <a
                    href={href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-medium text-blue-600 hover:underline"
                  >
                    {title}
                  </a>
                ) : (
                  <span className="font-medium">{title}</span>
                )}
                {href && (
                  <div className="text-xs text-slate-500">
                    {truncate(href, 100)}
                  </div>
                )}
                {snippet && <p className="mt-0.5 text-slate-700">{snippet}</p>}
                {pageAge && (
                  <div className="mt-0.5 text-xs text-slate-400">{pageAge}</div>
                )}
              </div>
            );
          })}
        </div>
      ),
    };
  }

  let content = getStringField(output, "content", "text", "_raw");
  if (!content) content = extractMcpText(output);
  if (!content) {
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
    title: statusCode ? `Response (${statusCode})` : "Web fetch",
    description: truncate(url, 80),
    content: content ? (
      <ContentCodeBlock>{content}</ContentCodeBlock>
    ) : message ? (
      <ContentMessage>{message}</ContentMessage>
    ) : Object.keys(output).length > 0 ? (
      <ContentCodeBlock>{JSON.stringify(output, null, 2)}</ContentCodeBlock>
    ) : null,
  };
}

function getBrowserAccordionData(
  toolName: string,
  input: unknown,
  output: Record<string, unknown>,
): AccordionData {
  const message = getStringField(output, "message");
  const snapshot = getStringField(output, "snapshot");

  // Screenshot tool: show the file_id so the user knows it was saved
  if (toolName === "browser_screenshot") {
    const fileId = getStringField(output, "file_id");
    const filename = getStringField(output, "filename");
    return {
      title: filename ? `Screenshot: ${filename}` : "Screenshot captured",
      description: fileId ? `file_id: ${fileId}` : undefined,
      content: message ? <ContentMessage>{message}</ContentMessage> : null,
    };
  }

  // Navigate / act tools: show snapshot if available
  const title =
    toolName === "browser_navigate"
      ? (getStringField(output, "title") ?? "Page loaded")
      : (message ?? "Action completed");

  const url = getStringField(output, "url", "current_url");

  return {
    title,
    description: url ? truncate(url, 80) : undefined,
    content: snapshot ? (
      <ContentCodeBlock>{truncate(snapshot, 3000)}</ContentCodeBlock>
    ) : message ? (
      <ContentMessage>{message}</ContentMessage>
    ) : null,
  };
}

function getFileAccordionData(
  category: ToolCategory,
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
  const mimeType = getStringField(output, "mime_type");
  const isImage = mimeType?.startsWith("image/");
  if (output.content_base64 && typeof output.content_base64 === "string") {
    if (isImage) {
      // Render image inline — handled below in the JSX
      displayContent = null;
    } else {
      try {
        const bytes = Uint8Array.from(atob(output.content_base64), (c) =>
          c.charCodeAt(0),
        );
        displayContent = new TextDecoder().decode(bytes);
      } catch {
        displayContent = "[Binary content]";
      }
    }
  }

  // Handle MCP-style content blocks from SDK tools (Read, Glob, Grep, Edit)
  if (!displayContent) {
    displayContent = extractMcpText(output);
  }

  // For edit: show old/new diff; for write: show written content if output is just a status
  const oldString =
    category === "edit"
      ? getStringField(inp as Record<string, unknown>, "old_string")
      : null;
  const newString =
    category === "edit"
      ? getStringField(inp as Record<string, unknown>, "new_string")
      : null;
  const writtenContent =
    category === "file-write"
      ? getStringField(inp as Record<string, unknown>, "content")
      : null;

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

  const isWriteOrEdit = category === "file-write" || category === "edit";

  return {
    title:
      message ??
      (isWriteOrEdit ? `Wrote ${truncate(filePath, 60)}` : "File output"),
    description: truncate(filePath, 80),
    content: (
      <div className="space-y-2">
        {oldString && newString != null ? (
          <>
            <div>
              <p className="mb-1 text-xs font-medium text-red-400">removed</p>
              <ContentCodeBlock>{oldString}</ContentCodeBlock>
            </div>
            <div>
              <p className="mb-1 text-xs font-medium text-green-400">added</p>
              <ContentCodeBlock>{newString}</ContentCodeBlock>
            </div>
          </>
        ) : writtenContent ? (
          <ContentCodeBlock>{writtenContent}</ContentCodeBlock>
        ) : isImage &&
          output.content_base64 &&
          typeof output.content_base64 === "string" ? (
          <RenderMedia
            value={`data:${mimeType};base64,${output.content_base64}`}
            metadata={{
              type: "image",
              mimeType: mimeType ?? undefined,
              filename: filePath ?? undefined,
            }}
          />
        ) : displayContent ? (
          <ContentCodeBlock>{displayContent}</ContentCodeBlock>
        ) : null}
        {fileListText && <ContentCodeBlock>{fileListText}</ContentCodeBlock>}
        {!displayContent && !fileListText && !writtenContent && message && (
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

function getAgentAccordionData(
  toolName: string,
  input: unknown,
  output: Record<string, unknown>,
): AccordionData {
  const inp = (input && typeof input === "object" ? input : {}) as Record<
    string,
    unknown
  >;
  const isAsync = output.isAsync === true || output.status === "async_launched";

  if (toolName === TOOL_TASK_OUTPUT) {
    const status = getStringField(output, "retrieval_status");
    const task = output.task;
    return {
      title: status === "timeout" ? "Agent still running" : "Agent result",
      description:
        typeof inp.agentId === "string" ? `Agent: ${inp.agentId}` : undefined,
      content: task ? (
        <ContentCodeBlock>{JSON.stringify(task, null, 2)}</ContentCodeBlock>
      ) : (
        <ContentMessage>
          {status === "timeout"
            ? "The agent hasn't finished yet. Results will appear automatically when it's done."
            : "No result available."}
        </ContentMessage>
      ),
    };
  }

  const description =
    getStringField(inp, "description") ?? getStringField(output, "description");
  const agentId = getStringField(output, "agentId");

  return {
    title: isAsync ? "Agent started (background)" : "Agent completed",
    description: description ?? agentId ?? undefined,
    content: isAsync ? (
      <ContentMessage>
        Running in the background. Results will appear here when ready.
      </ContentMessage>
    ) : (
      <ContentCodeBlock>{JSON.stringify(output, null, 2)}</ContentCodeBlock>
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
    content: <ContentCodeBlock>{displayContent}</ContentCodeBlock>,
  };
}

function getAccordionData(
  category: ToolCategory,
  toolName: string,
  input: unknown,
  output: Record<string, unknown>,
): AccordionData {
  switch (category) {
    case "bash":
      return getBashAccordionData(input, output);
    case "web":
      return getWebAccordionData(input, output);
    case "browser":
      return getBrowserAccordionData(toolName, input, output);
    case "file-read":
    case "file-write":
    case "file-delete":
    case "file-list":
    case "search":
    case "edit":
      return getFileAccordionData(category, input, output);
    case "todo":
      return getTodoAccordionData(input);
    case "agent":
      return getAgentAccordionData(toolName, input, output);
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

  // Compaction shows only a status line — no expandable accordion.
  const showAccordion =
    category !== "compaction" && (hasOutput || hasError || hasTodoInput);
  const accordionData = showAccordion
    ? getAccordionData(category, toolName, part.input, output ?? {})
    : null;

  return (
    <div className="py-2">
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
