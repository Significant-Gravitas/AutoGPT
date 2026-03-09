/**
 * Shared utilities for parsing and constructing workspace:// URIs.
 *
 * Format: workspace://{fileID}#{mimeType}
 *   - fileID: unique identifier for the file
 *   - mimeType: optional MIME type hint (e.g. "image/png")
 */

export interface WorkspaceURI {
  fileID: string;
  mimeType: string | null;
}

/**
 * Parse a workspace:// URI into its components.
 * Returns null if the string is not a workspace URI.
 */
export function parseWorkspaceURI(value: string): WorkspaceURI | null {
  if (!value.startsWith("workspace://")) return null;
  const rest = value.slice("workspace://".length);
  const hashIndex = rest.indexOf("#");
  if (hashIndex === -1) {
    return { fileID: rest, mimeType: null };
  }
  return {
    fileID: rest.slice(0, hashIndex),
    mimeType: rest.slice(hashIndex + 1) || null,
  };
}

/**
 * Extract just the file ID from a workspace:// URI.
 * Returns null if the string is not a workspace URI.
 */
export function parseWorkspaceFileID(uri: string): string | null {
  const parsed = parseWorkspaceURI(uri);
  return parsed?.fileID ?? null;
}

/**
 * Check if a value is a workspace:// URI string.
 */
export function isWorkspaceURI(value: unknown): value is string {
  return typeof value === "string" && value.startsWith("workspace://");
}

/**
 * Build a workspace:// URI from a file ID and optional MIME type.
 */
export function buildWorkspaceURI(fileID: string, mimeType?: string): string {
  return mimeType
    ? `workspace://${fileID}#${mimeType}`
    : `workspace://${fileID}`;
}
