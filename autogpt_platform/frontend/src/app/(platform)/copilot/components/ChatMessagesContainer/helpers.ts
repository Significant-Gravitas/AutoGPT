import { getGetWorkspaceDownloadFileByIdUrl } from "@/app/api/__generated__/endpoints/workspace/workspace";
import { ResponseType } from "@/app/api/__generated__/models/responseType";
import { parseWorkspaceURI } from "@/lib/workspace-uri";
import { FileUIPart, ToolUIPart, UIDataTypes, UIMessage, UITools } from "ai";
import type { ArtifactRef } from "../../store";

export type MessagePart = UIMessage<
  unknown,
  UIDataTypes,
  UITools
>["parts"][number];

export type RenderSegment =
  | { kind: "part"; part: MessagePart; index: number }
  | { kind: "collapsed-group"; parts: ToolUIPart[] };

const CUSTOM_TOOL_TYPES = new Set([
  "tool-ask_question",
  "tool-find_block",
  "tool-find_agent",
  "tool-find_library_agent",
  "tool-search_docs",
  "tool-get_doc_page",
  "tool-run_block",
  "tool-continue_run_block",
  "tool-run_mcp_tool",
  "tool-run_agent",
  "tool-schedule_agent",
  "tool-create_agent",
  "tool-edit_agent",
  "tool-view_agent_output",
  "tool-search_feature_requests",
  "tool-create_feature_request",
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

const WORKSPACE_FILE_PATTERN =
  /\/api\/proxy\/api\/workspace\/files\/([a-f0-9-]+)\/download/;
const WORKSPACE_URI_PATTERN = /workspace:\/\/([a-f0-9-]+)(?:#([^\s)\]]+))?/g;

const INTERACTIVE_RESPONSE_TYPES: ReadonlySet<string> = new Set([
  ResponseType.setup_requirements,
  ResponseType.agent_details,
  ResponseType.block_details,
  ResponseType.review_required,
  ResponseType.need_login,
  ResponseType.input_validation_error,
  ResponseType.agent_builder_clarification_needed,
  ResponseType.suggested_goal,
  ResponseType.agent_builder_preview,
  ResponseType.agent_builder_saved,
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
  let pendingGroup: Array<{ part: ToolUIPart; index: number }> | null = null;

  function flushGroup() {
    if (!pendingGroup) return;
    if (pendingGroup.length >= 2) {
      segments.push({
        kind: "collapsed-group",
        parts: pendingGroup.map((p) => p.part),
      });
    } else {
      for (const p of pendingGroup) {
        segments.push({ kind: "part", part: p.part, index: p.index });
      }
    }
    pendingGroup = null;
  }

  parts.forEach((part, i) => {
    const absoluteIndex = baseIndex + i;
    const isGenericCompletedTool =
      isCompletedToolPart(part) && !CUSTOM_TOOL_TYPES.has(part.type);

    if (isGenericCompletedTool) {
      if (!pendingGroup) pendingGroup = [];
      pendingGroup.push({ part: part as ToolUIPart, index: absoluteIndex });
    } else {
      flushGroup();
      segments.push({ kind: "part", part, index: absoluteIndex });
    }
  });

  flushGroup();
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
    if (isInteractiveToolPart(part)) {
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
): ArtifactRef | null {
  if (!file.url) return null;
  const match = file.url.match(WORKSPACE_FILE_PATTERN);
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

export function extractWorkspaceArtifacts(text: string): ArtifactRef[] {
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
      sourceUrl: `/api/proxy${getGetWorkspaceDownloadFileByIdUrl(parsed.fileID)}`,
      origin: "agent",
    });
  }

  return artifacts;
}

export function getMessageArtifacts(
  message: UIMessage<unknown, UIDataTypes, UITools>,
): ArtifactRef[] {
  const byId = new Map<string, ArtifactRef>();

  // Process file parts first — they carry richer metadata (mediaType from the
  // server, real filename) compared to workspace:// URIs extracted from text,
  // which often lack a MIME fragment and fall back to "File {id}".
  for (const part of message.parts) {
    if (part.type === "file") {
      const origin = message.role === "user" ? "user-upload" : "agent";
      const artifact = filePartToArtifactRef(part, origin);
      if (artifact) {
        byId.set(artifact.id, artifact);
      }
    }
  }

  for (const part of message.parts) {
    if (part.type === "text") {
      for (const artifact of extractWorkspaceArtifacts(part.text)) {
        if (!byId.has(artifact.id)) {
          byId.set(artifact.id, artifact);
        }
      }
    }
  }

  return Array.from(byId.values());
}

/**
 * Resolve workspace:// URLs in markdown text to proxy download URLs.
 *
 * Handles both image syntax  `![alt](workspace://id#mime)` and regular link
 * syntax `[text](workspace://id)`.  For images the MIME type hash fragment is
 * inspected so that videos can be rendered with a `<video>` element via the
 * custom img component.
 */
export function resolveWorkspaceUrls(text: string): string {
  // Handle image links: ![alt](workspace://id#mime)
  let resolved = text.replace(
    /!\[([^\]]*)\]\(workspace:\/\/([^)#\s]+)(?:#([^)#\s]*))?\)/g,
    (_match, alt: string, fileId: string, mimeHint?: string) => {
      const apiPath = getGetWorkspaceDownloadFileByIdUrl(fileId);
      const url = `/api/proxy${apiPath}`;
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
      const apiPath = getGetWorkspaceDownloadFileByIdUrl(fileId);
      const origin =
        typeof window !== "undefined" ? window.location.origin : "";
      const url = `${origin}/api/proxy${apiPath}`;
      return `[${linkText || "Download file"}](${url})`;
    },
  );

  return resolved;
}
