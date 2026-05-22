// Single source of truth for every share-related path: the viewer
// route that ends up on a user's clipboard, and the API path the
// frontend hits when toggling shares.  When the next shareable type
// joins the platform, add functions here and grep won't find anything
// else to touch.

function getBaseUrl(): string {
  // ``NEXT_PUBLIC_FRONTEND_BASE_URL`` lets the backend's share URL and
  // the frontend's share URL match in environments where Next.js is
  // proxied behind a different host than its window.location.  Falls
  // back to the current origin in dev.
  if (typeof window === "undefined") {
    return process.env.NEXT_PUBLIC_FRONTEND_BASE_URL || "";
  }
  return process.env.NEXT_PUBLIC_FRONTEND_BASE_URL || window.location.origin;
}

export function executionSharePath(token: string): string {
  return `/share/${token}`;
}

export function chatSharePath(token: string): string {
  return `/share/chat/${token}`;
}

export function executionShareUrl(token: string): string {
  return `${getBaseUrl()}${executionSharePath(token)}`;
}

export function chatShareUrl(token: string): string {
  return `${getBaseUrl()}${chatSharePath(token)}`;
}

// ---------- Per-share file download URL + matcher --------------------------
//
// The renderer extracts a file ID from a ``FileUIPart.url`` to decide
// whether to render an ArtifactCard.  Keeping the URL builder and the
// matching regex in the same module guarantees they evolve together —
// if the public-share file route ever changes, both update here and
// nowhere else needs to be touched.

const FILE_UUID_GROUP =
  "([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})";

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function sharedChatFileUrl(shareToken: string, fileId: string): string {
  return `/api/proxy/api/public/shared/chats/${shareToken}/files/${fileId}/download`;
}

export function sharedChatFilePattern(shareToken: string): RegExp {
  // Anchor on the literal share-chat prefix for THIS token so only
  // URLs we constructed for this specific share match.  Per-token
  // patterns prevent cross-share contamination when multiple viewers
  // are mounted side-by-side (storybook / tests).
  //
  // ``^`` and ``$`` enforce full-string match — without them the regex
  // could pull a file UUID out of a longer/malformed URL (e.g.
  // ``https://attacker.example/api/proxy/.../files/<id>/download?x=…``)
  // and surface it as an artifact for this share token even though we
  // never constructed that URL.
  return new RegExp(
    `^/api/proxy/api/public/shared/chats/${escapeRegex(shareToken)}/files/${FILE_UUID_GROUP}/download$`,
  );
}
