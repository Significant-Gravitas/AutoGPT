import { getGetWorkspaceDownloadFileByIdUrl } from "@/app/api/__generated__/endpoints/workspace/workspace";

// Special message prefixes for text-based markers (set by backend).
// The hex suffix makes it virtually impossible for an LLM to accidentally
// produce these strings in normal conversation.
const COPILOT_ERROR_PREFIX = "[__COPILOT_ERROR_f7a1__]";
const COPILOT_SYSTEM_PREFIX = "[__COPILOT_SYSTEM_e3b0__]";

export type MarkerType = "error" | "system" | null;

/** Escape all regex special characters in a string. */
function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// Pre-compiled marker regexes (avoids re-creating on every call / render)
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
