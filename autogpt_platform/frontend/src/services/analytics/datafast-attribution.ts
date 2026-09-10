const COOKIE_TO_HEADER = {
  datafast_visitor_id: "X-Datafast-Visitor-Id",
  datafast_session_id: "X-Datafast-Session-Id",
} as const;

// Reads the DataFast attribution cookies (set client-side by the DataFast
// script) and maps them to request headers. Best-effort: returns only the
// cookies that exist, and {} during SSR. Never throws.
export function getDatafastAttribution(): Record<string, string> {
  if (typeof document === "undefined") return {};

  const jar = new Map(
    document.cookie.split("; ").map((part) => {
      const eq = part.indexOf("=");
      return eq === -1 ? [part, ""] : [part.slice(0, eq), part.slice(eq + 1)];
    }) as [string, string][],
  );

  const headers: Record<string, string> = {};
  for (const [cookie, header] of Object.entries(COOKIE_TO_HEADER)) {
    const rawValue = jar.get(cookie);
    if (!rawValue) continue;
    try {
      headers[header] = decodeURIComponent(rawValue);
    } catch {
      // Best-effort: ignore a malformed percent-encoded value rather than throw.
    }
  }
  return headers;
}
