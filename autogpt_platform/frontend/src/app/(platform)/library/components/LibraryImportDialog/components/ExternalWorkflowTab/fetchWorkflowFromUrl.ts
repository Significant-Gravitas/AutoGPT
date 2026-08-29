"use server";

/**
 * Regex to extract the numeric template ID from various n8n URL formats:
 *   - https://n8n.io/workflows/1234
 *   - https://n8n.io/workflows/1234-some-slug
 *   - https://api.n8n.io/api/templates/workflows/1234
 */
const N8N_TEMPLATE_ID_RE = /n8n\.io\/(?:api\/templates\/)?workflows\/(\d+)/i;

/** Hardcoded n8n templates API base — the only URL we ever fetch. */
const N8N_TEMPLATES_API = "https://api.n8n.io/api/templates/workflows";

/** Max response body size (10 MB) to prevent memory exhaustion. */
const MAX_RESPONSE_BYTES = 10 * 1024 * 1024;

export type FetchWorkflowResult =
  | { ok: true; json: string }
  | { ok: false; error: string };

/**
 * Server action that fetches a workflow JSON from an n8n template URL.
 * Runs server-side so there are no CORS restrictions.
 *
 * Returns a result object instead of throwing because Next.js
 * server actions do not propagate error messages to the client.
 *
 * Only n8n.io workflow URLs are accepted. The template ID is extracted
 * and used to call the hardcoded n8n API — the user-supplied URL is
 * never passed to fetch() directly (SSRF prevention).
 */
export async function fetchWorkflowFromUrl(
  url: string,
): Promise<FetchWorkflowResult> {
  const match = url.match(N8N_TEMPLATE_ID_RE);
  if (!match) {
    return {
      ok: false,
      error:
        "Invalid or unsupported URL. " +
        "URL import is supported for n8n.io workflow templates " +
        "(e.g. https://n8n.io/workflows/1234). " +
        "For other platforms, use file upload.",
    };
  }

  const templateId = match[1]; // purely numeric, safe to interpolate

  try {
    const json = await fetchN8nWorkflow(templateId);
    return { ok: true, json };
  } catch (err) {
    return {
      ok: false,
      error: err instanceof Error ? err.message : "Failed to fetch workflow.",
    };
  }
}

async function fetchN8nWorkflow(templateId: string): Promise<string> {
  // Only ever fetch from the hardcoded API base + numeric ID.
  // parseInt + toString round-trips to guarantee the value is purely numeric,
  // preventing any path-traversal or SSRF via the interpolated segment.
  const safeId = parseInt(templateId, 10);
  if (!Number.isFinite(safeId) || safeId <= 0) {
    throw new Error("Invalid template ID");
  }
  const res = await fetch(`${N8N_TEMPLATES_API}/${safeId.toString()}`);
  if (!res.ok) throw new Error(`n8n template not found (${res.status})`);

  const contentLength = res.headers.get("content-length");
  if (contentLength && parseInt(contentLength, 10) > MAX_RESPONSE_BYTES) {
    throw new Error("Response too large.");
  }

  const text = await res.text();
  if (text.length > MAX_RESPONSE_BYTES) throw new Error("Response too large.");

  const data = JSON.parse(text);
  const template = data?.workflow ?? data;
  const workflow = template?.workflow ?? template;
  if (!workflow?.nodes) throw new Error("Unexpected n8n API response format");
  if (!workflow.name) workflow.name = template?.name ?? data?.name ?? "";
  return JSON.stringify(workflow);
}
