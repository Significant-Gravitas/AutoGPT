import { getGetWorkspaceDownloadFileByIdUrl } from "@/app/api/__generated__/endpoints/workspace/workspace";
import { ResponseType } from "@/app/api/__generated__/models/responseType";
import { parseWorkspaceURI } from "@/lib/workspace-uri";
import { FileUIPart, ToolUIPart, UIDataTypes, UIMessage, UITools } from "ai";
import { isCorruptedCardToolPart } from "../../helpers/toolOutput";
import type { ArtifactRef } from "../../store";
import type { TodoItem } from "../ContextPanel/components/ProgressTab/helpers";

export function shouldShowTaskListNotice({
  isContextPanelEnabled,
  isChatStreaming,
  latestTaskList,
}: {
  isContextPanelEnabled: boolean;
  isChatStreaming: boolean;
  latestTaskList: TodoItem[] | null;
}): boolean {
  if (!isContextPanelEnabled || !isChatStreaming || !latestTaskList) {
    return false;
  }
  return latestTaskList.some((t) => t.status !== "completed");
}

export type MessagePart = UIMessage<
  unknown,
  UIDataTypes,
  UITools
>["parts"][number];

export type RenderSegment =
  | { kind: "part"; part: MessagePart; index: number }
  | { kind: "collapsed-group"; parts: ToolUIPart[] }
  | { kind: "reasoning-group"; parts: MessagePart[]; index: number };

const CUSTOM_TOOL_TYPES = new Set([
  "tool-ask_question",
  "tool-find_block",
  "tool-find_agent",
  "tool-find_library_agent",
  "tool-search_docs",
  "tool-get_doc_page",
  "tool-connect_integration",
  "tool-run_block",
  "tool-continue_run_block",
  "tool-connect_integration",
  "tool-run_mcp_tool",
  "tool-run_agent",
  "tool-schedule_agent",
  "tool-setup_agent_webhook_trigger",
  "tool-create_agent",
  "tool-edit_agent",
  "tool-view_agent_output",
  "tool-search_feature_requests",
  "tool-create_feature_request",
  "tool-decompose_goal",
]);

const REASONING_TOOL_TYPES = new Set([
  "tool-find_block",
  "tool-find_agent",
  "tool-find_library_agent",
  "tool-search_docs",
  "tool-get_doc_page",
  "tool-search_feature_requests",
  "tool-ask_question",
]);

export function isReasoningToolPart(part: MessagePart): boolean {
  return REASONING_TOOL_TYPES.has(part.type);
}

// Default workspace-file URL shape: ``/api/proxy/api/workspace/files/<uuid>/download``.
// Other surfaces (e.g. public share viewer) pass their own pattern into
// ``filePartToArtifactRef`` rather than loosen this one — keeping the
// match anchored to a known prefix per surface prevents an unrelated
// future ``FileUIPart`` source from accidentally rendering as an
// artifact.  ``^`` and ``$`` are required — without them, the pattern
// matches as a substring inside longer URLs (e.g. an attacker-controlled
// file URL prefixed with the proxy path) and surfaces the embedded UUID
// as a renderable artifact id.
export const WORKSPACE_FILE_PATTERN =
  /^\/api\/proxy\/api\/workspace\/files\/([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})\/download$/;
const WORKSPACE_URI_PATTERN = /workspace:\/\/([a-f0-9-]+)(?:#([^\s)\]]+))?/g;

const INTERACTIVE_RESPONSE_TYPES: ReadonlySet<string> = new Set([
  ResponseType.setup_requirements,
  ResponseType.trigger_setup,
  ResponseType.agent_details,
  ResponseType.block_details,
  ResponseType.review_required,
  ResponseType.need_login,
  ResponseType.input_validation_error,
  ResponseType.agent_builder_clarification_needed,
  ResponseType.suggested_goal,
  ResponseType.agent_builder_preview,
  ResponseType.agent_builder_saved,
  ResponseType.task_decomposition,
]);

export function isCompletedToolPart(part: MessagePart): part is ToolUIPart {
  return (
    part.type.startsWith("tool-") &&
    "state" in part &&
    (part.state === "output-available" || part.state === "output-error")
  );
}

export function isInteractiveToolPart(part: MessagePart): boolean {
  if (!part.type.startsWith("tool-")) return false;
  if (!("state" in part) || part.state !== "output-available") return false;

  let output = (part as ToolUIPart).output;
  if (!output) return false;

  if (typeof output === "string") {
    try {
      output = JSON.parse(output);
    } catch {
      return false;
    }
  }

  if (typeof output !== "object" || output === null) return false;

  const responseType = (output as Record<string, unknown>).type;
  return (
    typeof responseType === "string" &&
    INTERACTIVE_RESPONSE_TYPES.has(responseType)
  );
}

export function buildRenderSegments(
  parts: MessagePart[],
  baseIndex = 0,
): RenderSegment[] {
  const segments: RenderSegment[] = [];
  let pendingTools: Array<{ part: ToolUIPart; index: number }> | null = null;
  let pendingReasoning: Array<{ part: MessagePart; index: number }> | null =
    null;

  function flushTools() {
    if (!pendingTools) return;
    if (pendingTools.length >= 2) {
      segments.push({
        kind: "collapsed-group",
        parts: pendingTools.map((p) => p.part),
      });
    } else {
      for (const p of pendingTools) {
        segments.push({ kind: "part", part: p.part, index: p.index });
      }
    }
    pendingTools = null;
  }

  // Native reasoning parts (one per agentic turn) are always emitted as a
  // reasoning-group — including a lone part. This folds a multi-step task's
  // consecutive reasoning into one collapsed block instead of a stacked wall of
  // "Reasoning" accordions, and gives the run a single stable identity
  // (`reasoning-group` keyed by its first index) so a single block doesn't
  // remount — losing its open/closed state — the moment a second consecutive
  // block arrives and turns it into a group.
  function flushReasoning() {
    if (!pendingReasoning) return;
    segments.push({
      kind: "reasoning-group",
      parts: pendingReasoning.map((p) => p.part),
      index: pendingReasoning[0].index,
    });
    pendingReasoning = null;
  }

  parts.forEach((part, i) => {
    const absoluteIndex = baseIndex + i;

    // `step-start` markers delimit turns in multi-step agentic streams and
    // render as nothing. Treat them as transparent: skipping them (rather than
    // flushing) keeps the reasoning blocks on either side in a single run, which
    // is exactly the consecutive-reasoning case this grouping targets.
    if (part.type === "step-start") return;

    const isGenericCompletedTool =
      isCompletedToolPart(part) && !CUSTOM_TOOL_TYPES.has(part.type);

    if (isGenericCompletedTool) {
      flushReasoning();
      if (!pendingTools) pendingTools = [];
      pendingTools.push({ part: part as ToolUIPart, index: absoluteIndex });
    } else if (part.type === "reasoning") {
      flushTools();
      if (!pendingReasoning) pendingReasoning = [];
      pendingReasoning.push({ part, index: absoluteIndex });
    } else {
      flushTools();
      flushReasoning();
      segments.push({ kind: "part", part, index: absoluteIndex });
    }
  });

  flushTools();
  flushReasoning();
  return segments;
}

function isReasoningBoundary(part: MessagePart): boolean {
  return part.type === "reasoning" || isReasoningToolPart(part);
}

export function splitReasoningAndResponse(parts: MessagePart[]): {
  reasoning: MessagePart[];
  response: MessagePart[];
} {
  const lastReasoningIndex = parts.findLastIndex(isReasoningBoundary);

  if (lastReasoningIndex === -1) {
    return { reasoning: [], response: parts };
  }

  const hasResponseAfterReasoning = parts
    .slice(lastReasoningIndex + 1)
    .some((p) => p.type === "text");

  if (!hasResponseAfterReasoning) {
    return { reasoning: [], response: parts };
  }

  const rawReasoning = parts.slice(0, lastReasoningIndex + 1);
  const rawResponse = parts.slice(lastReasoningIndex + 1);

  const reasoning: MessagePart[] = [];
  const pinnedParts: MessagePart[] = [];

  for (const part of rawReasoning) {
    // Corrupted card-capable parts are pinned too: their output failed to
    // parse, so isInteractiveToolPart can't recognize them, but hiding them
    // in the steps modal would silently swallow a lost sign-in/setup card.
    // Pinning lets the tool renderer surface a visible error instead.
    if (isInteractiveToolPart(part) || isCorruptedCardToolPart(part)) {
      pinnedParts.push(part);
    } else {
      // Reasoning / thinking parts stay inside the outer "Show steps" modal
      // alongside the tool-use timeline — their own inline accordion handles
      // expansion inside the modal so there's no visual collision.
      reasoning.push(part);
    }
  }

  return {
    reasoning,
    response: [...pinnedParts, ...rawResponse],
  };
}

export function getTurnMessages(
  messages: UIMessage<unknown, UIDataTypes, UITools>[],
  lastAssistantIndex: number,
): UIMessage<unknown, UIDataTypes, UITools>[] {
  const userIndex = messages.findLastIndex(
    (m, i) => i < lastAssistantIndex && m.role === "user",
  );
  const nextUserIndex = messages.findIndex(
    (m, i) => i > lastAssistantIndex && m.role === "user",
  );
  const start = userIndex >= 0 ? userIndex : lastAssistantIndex;
  const end = nextUserIndex >= 0 ? nextUserIndex : messages.length;
  return messages.slice(start, end);
}

// Special message prefixes for text-based markers (set by backend).
// The hex suffix makes it virtually impossible for an LLM to accidentally
// produce these strings in normal conversation.
const COPILOT_ERROR_PREFIX = "[__COPILOT_ERROR_f7a1__]";
const COPILOT_RETRYABLE_ERROR_PREFIX = "[__COPILOT_RETRYABLE_ERROR_a9c2__]";
const COPILOT_SYSTEM_PREFIX = "[__COPILOT_SYSTEM_e3b0__]";

export type MarkerType = "error" | "retryable_error" | "system" | null;

/** Escape all regex special characters in a string. */
function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// Pre-compiled marker regexes (avoids re-creating on every call / render).
// Retryable check must come first since it's more specific.
const RETRYABLE_ERROR_MARKER_RE = new RegExp(
  `${escapeRegExp(COPILOT_RETRYABLE_ERROR_PREFIX)}\\s*(.+?)$`,
  "s",
);
const ERROR_MARKER_RE = new RegExp(
  `${escapeRegExp(COPILOT_ERROR_PREFIX)}\\s*(.+?)$`,
  "s",
);
const SYSTEM_MARKER_RE = new RegExp(
  `${escapeRegExp(COPILOT_SYSTEM_PREFIX)}\\s*(.+?)$`,
  "s",
);

export function parseSpecialMarkers(text: string): {
  markerType: MarkerType;
  markerText: string;
  cleanText: string;
} {
  const retryableMatch = text.match(RETRYABLE_ERROR_MARKER_RE);
  if (retryableMatch) {
    return {
      markerType: "retryable_error",
      markerText: retryableMatch[1].trim(),
      cleanText: text.replace(retryableMatch[0], "").trim(),
    };
  }

  const errorMatch = text.match(ERROR_MARKER_RE);
  if (errorMatch) {
    return {
      markerType: "error",
      markerText: errorMatch[1].trim(),
      cleanText: text.replace(errorMatch[0], "").trim(),
    };
  }

  const systemMatch = text.match(SYSTEM_MARKER_RE);
  if (systemMatch) {
    return {
      markerType: "system",
      markerText: systemMatch[1].trim(),
      cleanText: text.replace(systemMatch[0], "").trim(),
    };
  }

  return { markerType: null, markerText: "", cleanText: text };
}

export function filePartToArtifactRef(
  file: FileUIPart,
  origin: ArtifactRef["origin"] = "user-upload",
  /** Pattern that extracts the file UUID from ``file.url``.  Defaults
   *  to the workspace-file shape; the public share viewer passes a
   *  per-token pattern from ``lib/share/routes.ts`` so its file URLs
   *  match without loosening the default. */
  pattern: RegExp = WORKSPACE_FILE_PATTERN,
): ArtifactRef | null {
  if (!file.url) return null;
  const match = file.url.match(pattern);
  if (!match) return null;
  return {
    id: match[1],
    title: file.filename || "File",
    mimeType: file.mediaType || null,
    sourceUrl: file.url,
    origin,
  };
}

const FULL_UUID =
  /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/;

/** Build the default workspace-file URL — used wherever a caller
 *  doesn't supply its own ``fileUrlBuilder``.  Centralising it here
 *  keeps the owner-side default in one place. */
function defaultWorkspaceFileUrl(fileId: string): string {
  return `/api/proxy${getGetWorkspaceDownloadFileByIdUrl(fileId)}`;
}

export function extractWorkspaceArtifacts(
  text: string,
  fileUrlBuilder: (fileId: string) => string = defaultWorkspaceFileUrl,
): ArtifactRef[] {
  const seen = new Set<string>();
  const artifacts: ArtifactRef[] = [];

  for (const match of text.matchAll(WORKSPACE_URI_PATTERN)) {
    const fullUri = match[0];
    const parsed = parseWorkspaceURI(fullUri);

    if (!parsed || seen.has(parsed.fileID)) continue;

    // During streaming, workspace:// URIs arrive character-by-character.
    // The regex matches progressively longer partial IDs — reject them so
    // ArtifactCards don't mount/unmount with garbage IDs.
    if (!FULL_UUID.test(parsed.fileID)) continue;

    // Skip URIs inside image markdown (`![alt](workspace://...)`). Images are
    // rendered inline via resolveWorkspaceUrls — surfacing them as cards too
    // would double-render the same asset.
    const escapedUri = escapeRegExp(fullUri);
    const imagePattern = new RegExp(`!\\[[^\\]]*\\]\\(${escapedUri}\\)`);
    if (imagePattern.test(text)) continue;

    seen.add(parsed.fileID);

    const linkPattern = new RegExp(`\\[([^\\]]+)\\]\\(${escapedUri}\\)`);
    const linkMatch = text.match(linkPattern);
    const title = linkMatch?.[1] ?? `File ${parsed.fileID.slice(0, 8)}`;

    artifacts.push({
      id: parsed.fileID,
      title,
      mimeType: parsed.mimeType,
      sourceUrl: fileUrlBuilder(parsed.fileID),
      origin: "agent",
    });
  }

  return artifacts;
}

export function getMessageArtifacts(
  message: UIMessage<unknown, UIDataTypes, UITools>,
  options: {
    filePattern?: RegExp;
    fileUrlBuilder?: (fileId: string) => string;
  } = {},
): ArtifactRef[] {
  const byId = new Map<string, ArtifactRef>();

  // Process file parts first — they carry richer metadata (mediaType from the
  // server, real filename) compared to workspace:// URIs extracted from text,
  // which often lack a MIME fragment and fall back to "File {id}".
  for (const part of message.parts) {
    if (part.type === "file") {
      const origin = message.role === "user" ? "user-upload" : "agent";
      const artifact = filePartToArtifactRef(part, origin, options.filePattern);
      if (artifact) {
        byId.set(artifact.id, artifact);
      }
    }
  }

  for (const part of message.parts) {
    if (part.type === "text") {
      for (const artifact of extractWorkspaceArtifacts(
        part.text,
        options.fileUrlBuilder,
      )) {
        if (!byId.has(artifact.id)) {
          byId.set(artifact.id, artifact);
        }
      }
    }
  }

  return Array.from(byId.values());
}

export function getMostRecentArtifact(
  messages: UIMessage<unknown, UIDataTypes, UITools>[],
  options: {
    filePattern?: RegExp;
    fileUrlBuilder?: (fileId: string) => string;
    origin?: ArtifactRef["origin"];
  } = {},
): ArtifactRef | null {
  for (
    let messageIndex = messages.length - 1;
    messageIndex >= 0;
    messageIndex--
  ) {
    const message = messages[messageIndex];
    for (
      let partIndex = message.parts.length - 1;
      partIndex >= 0;
      partIndex--
    ) {
      const part = message.parts[partIndex];
      if (part.type === "file") {
        const origin = message.role === "user" ? "user-upload" : "agent";
        const artifact = filePartToArtifactRef(
          part,
          origin,
          options.filePattern,
        );
        if (
          artifact &&
          (!options.origin || artifact.origin === options.origin)
        ) {
          return artifact;
        }
      }
      if (part.type === "text") {
        const artifacts = extractWorkspaceArtifacts(
          part.text,
          options.fileUrlBuilder,
        );
        for (
          let artifactIndex = artifacts.length - 1;
          artifactIndex >= 0;
          artifactIndex--
        ) {
          const artifact = artifacts[artifactIndex];
          if (!options.origin || artifact.origin === options.origin) {
            return artifact;
          }
        }
      }
    }
  }
  return null;
}

/**
 * Resolve workspace:// URLs in markdown text to proxy download URLs.
 *
 * Handles both image syntax  `![alt](workspace://id#mime)` and regular link
 * syntax `[text](workspace://id)`.  For images the MIME type hash fragment is
 * inspected so that videos can be rendered with a `<video>` element via the
 * custom img component.
 */
export function resolveWorkspaceUrls(
  text: string,
  fileUrlBuilder: (fileId: string) => string = defaultWorkspaceFileUrl,
): string {
  // Handle image links: ![alt](workspace://id#mime)
  let resolved = text.replace(
    /!\[([^\]]*)\]\(workspace:\/\/([^)#\s]+)(?:#([^)#\s]*))?\)/g,
    (_match, alt: string, fileId: string, mimeHint?: string) => {
      const url = fileUrlBuilder(fileId);
      if (mimeHint?.startsWith("video/")) {
        return `![video:${alt || "Video"}](${url})`;
      }
      return `![${alt || "Image"}](${url})`;
    },
  );

  // Handle regular links: [text](workspace://id) — without the leading "!"
  // These are blocked by Streamdown's rehype-harden sanitizer because
  // "workspace://" is not in the allowed URL-scheme whitelist, which causes
  // "[blocked]" to appear next to the link text.
  // Use an absolute URL so Streamdown's "Copy link" button copies the full
  // URL (including host) rather than just the path.
  resolved = resolved.replace(
    /(?<!!)\[([^\]]*)\]\(workspace:\/\/([^)#\s]+)(?:#[^)#\s]*)?\)/g,
    (_match, linkText: string, fileId: string) => {
      const url = fileUrlBuilder(fileId);
      const origin =
        typeof window !== "undefined" ? window.location.origin : "";
      const absoluteUrl = url.startsWith("/") ? `${origin}${url}` : url;
      return `[${linkText || "Download file"}](${absoluteUrl})`;
    },
  );

  return resolved;
}
