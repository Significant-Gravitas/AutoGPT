"use server";

const N8N_URL_RE = /n8n\.io\/workflows\/(\d+)/i;
const N8N_TEMPLATES_API = "https://api.n8n.io/api/templates/workflows";

/** Hostnames allowed for URL-based workflow import (SSRF prevention). */
const ALLOWED_HOSTS = ["n8n.io", "api.n8n.io"];

/**
 * Server action that fetches a workflow JSON from a URL.
 * Runs server-side so there are no CORS restrictions.
 *
 * Only URLs from known workflow platform hosts are accepted
 * to prevent SSRF. Currently supports n8n.io workflows.
 */
export async function fetchWorkflowFromUrl(url: string): Promise<string> {
  // Validate host against allowlist to prevent SSRF
  let hostname: string;
  try {
    hostname = new URL(url).hostname;
  } catch {
    throw new Error("Invalid URL");
  }

  if (
    !ALLOWED_HOSTS.some((h) => hostname === h || hostname.endsWith(`.${h}`))
  ) {
    throw new Error(
      "Unsupported host. URL import is supported for n8n.io workflows. " +
        "For other platforms, use file upload.",
    );
  }

  const n8nMatch = url.match(N8N_URL_RE);
  if (n8nMatch) {
    return fetchN8nWorkflow(n8nMatch[1]);
  }

  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch workflow (${res.status})`);
  }
  const json = await res.text();
  JSON.parse(json); // validate
  return json;
}

async function fetchN8nWorkflow(templateId: string): Promise<string> {
  const res = await fetch(`${N8N_TEMPLATES_API}/${templateId}`);
  if (!res.ok) throw new Error(`n8n template not found (${res.status})`);

  const data = await res.json();
  // n8n API: { workflow: { workflow: { nodes, connections, ... }, name, ... } }
  const template = data?.workflow ?? data;
  const workflow = template?.workflow ?? template;
  if (!workflow?.nodes) throw new Error("Unexpected n8n API response format");
  if (!workflow.name) workflow.name = template?.name ?? data?.name ?? "";
  return JSON.stringify(workflow);
}
