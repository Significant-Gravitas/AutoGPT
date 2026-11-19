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
