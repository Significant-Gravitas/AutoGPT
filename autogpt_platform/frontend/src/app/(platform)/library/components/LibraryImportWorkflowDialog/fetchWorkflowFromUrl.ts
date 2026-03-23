"use server";

const N8N_URL_RE = /n8n\.io\/workflows\/(\d+)/i;
const N8N_TEMPLATES_API = "https://api.n8n.io/api/templates/workflows";

/** Hostnames allowed for URL-based workflow import (SSRF prevention). */
const ALLOWED_HOSTS = ["n8n.io", "api.n8n.io"];

export type FetchWorkflowResult =
  | { ok: true; json: string }
  | { ok: false; error: string };

/**
 * Server action that fetches a workflow JSON from a URL.
 * Runs server-side so there are no CORS restrictions.
 *
 * Returns a result object instead of throwing because Next.js
 * server actions do not propagate error messages to the client.
 *
 * Only URLs from known workflow platform hosts are accepted
 * to prevent SSRF. Currently supports n8n.io workflows.
 */
export async function fetchWorkflowFromUrl(
  url: string,
): Promise<FetchWorkflowResult> {
  let hostname: string;
  try {
    hostname = new URL(url).hostname;
  } catch {
    return { ok: false, error: "Invalid URL." };
  }

  if (
    !ALLOWED_HOSTS.some((h) => hostname === h || hostname.endsWith(`.${h}`))
  ) {
    return {
      ok: false,
      error:
        "Unsupported host. URL import is supported for n8n.io workflows. " +
        "For other platforms, use file upload.",
    };
  }

  try {
    const n8nMatch = url.match(N8N_URL_RE);
    const json = n8nMatch
      ? await fetchN8nWorkflow(n8nMatch[1])
      : await fetchGenericJson(url);
    return { ok: true, json };
  } catch (err) {
    return {
      ok: false,
      error: err instanceof Error ? err.message : "Failed to fetch workflow.",
    };
  }
}

async function fetchGenericJson(url: string): Promise<string> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch workflow (${res.status})`);
  const text = await res.text();
  JSON.parse(text); // validate
  return text;
}

async function fetchN8nWorkflow(templateId: string): Promise<string> {
  const res = await fetch(`${N8N_TEMPLATES_API}/${templateId}`);
  if (!res.ok) throw new Error(`n8n template not found (${res.status})`);

  const data = await res.json();
  const template = data?.workflow ?? data;
  const workflow = template?.workflow ?? template;
  if (!workflow?.nodes) throw new Error("Unexpected n8n API response format");
  if (!workflow.name) workflow.name = template?.name ?? data?.name ?? "";
  return JSON.stringify(workflow);
}
