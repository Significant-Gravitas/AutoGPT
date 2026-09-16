import { getGetWorkspaceDownloadFileByIdUrl } from "@/app/api/__generated__/endpoints/workspace/workspace";
import { parseWorkspaceURI } from "@/lib/workspace-uri";
import { FileUIPart, ToolUIPart, UIDataTypes, UIMessage, UITools } from "ai";
import type { ArtifactRef } from "../../store";
import {
  COMPACTION_DATA_PART_TYPE,
  readCompactionStats,
  type CompactionPhase,
  type CompactionStats,
} from "../CompactionCard/helpers";
import { COMPACTION_PART_TYPE } from "../ToolChain/helpers";

export type MessagePart = UIMessage<
  unknown,
  UIDataTypes,
  UITools
>["parts"][number];

// Every assistant tool renders inside the ToolChain. ToolResult supplies a
// compact result view for known backend tools and a structured fallback for
// SDK or future tools, so no tool ever renders as a bare top-level part.
export function isChainableToolPart(part: MessagePart): boolean {
  if (part.type === COMPACTION_PART_TYPE) return false;
  return part.type === "reasoning" || part.type.startsWith("tool-");
}

const COMPACTION_PHASES = new Set(["summarizing", "rebuilding"]);

// All `data-*` parts are transient bookkeeping (status, cursor,
// pending-drained, mode-changed, …) — none of them is content that settles
// a compaction row, and neither may any future one. Enumerating them here
// would silently kill the bar the day a new data part ships mid-compaction.
function isCompactionTransparentPart(part: MessagePart): boolean {
  return (
    part.type.startsWith("data-") ||
    part.type === COMPACTION_PART_TYPE ||
    part.type === "step-start"
  );
}

/**
 * A row closed by the abort sentinel (output "") or an error never compacted
 * anything — any phase parts it left behind are stale, and the row itself
 * renders nothing. This is a cross-language contract with the backend's
 * close paths, so it lives here once: `MessagePartRenderer` decides what to
 * draw with the same predicate `getLatestCompactionPhase` decides with.
 */
export function isRetiredCompactionRow(part: MessagePart): boolean {
  if (part.type !== COMPACTION_PART_TYPE || !("state" in part)) return false;
  const tool = part as ToolUIPart;
  if (tool.state === "output-error") return true;
  return (
    tool.state === "output-available" &&
    typeof tool.output === "string" &&
    tool.output.trim() === ""
  );
}

/**
 * Latest `data-compaction` phase on a message, or null once real content has
 * landed past it (at which point the compaction row is settled history).
 */
export function getLatestCompactionPhase(
  parts: MessagePart[],
): CompactionPhase | null {
  const lastRow = parts.findLast((p) => p.type === COMPACTION_PART_TYPE);
  if (lastRow && isRetiredCompactionRow(lastRow)) return null;
  for (let i = parts.length - 1; i >= 0; i--) {
    const part = parts[i];
    if (part.type === COMPACTION_DATA_PART_TYPE) {
      const data = (part as { data?: { phase?: unknown } }).data;
      const phase = data?.phase;
      if (typeof phase === "string" && COMPACTION_PHASES.has(phase)) {
        return phase as CompactionPhase;
      }
      return null;
    }
    if (isCompactionTransparentPart(part)) continue;
    if (part.type === "text" && "text" in part && !part.text.trim()) continue;
    return null;
  }
  return null;
}

/**
 * Stats carried by the message's `data-compaction` parts, merged in stream
 * order (later phases override earlier ones). They pace the live progress
 * curve before the tool row closes with its own — authoritative — stats.
 */
export function getLatestCompactionStats(
  parts: MessagePart[],
): CompactionStats {
  const stats: CompactionStats = {};
  for (const part of parts) {
    if (part.type !== COMPACTION_DATA_PART_TYPE) continue;
    const data = (part as { data?: unknown }).data;
    if (typeof data !== "object" || data === null) continue;
    Object.assign(stats, readCompactionStats(data as Record<string, unknown>));
  }
  return stats;
}

/**
 * Tool-call ID of the message's last `tool-context_compaction` part. The live
 * phase applies only to this row — without the ID gate, a second compaction
 * cycle's `summarizing` part would flip earlier (settled) rows back to live.
 */
export function getLastCompactionCallId(parts: MessagePart[]): string | null {
  for (let i = parts.length - 1; i >= 0; i--) {
    const part = parts[i];
    if (part.type === COMPACTION_PART_TYPE && "toolCallId" in part) {
      return (part as ToolUIPart).toolCallId ?? null;
    }
  }
  return null;
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
