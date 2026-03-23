"use server";

const N8N_URL_RE = /n8n\.io\/workflows\/(\d+)/i;
const N8N_TEMPLATES_API = "https://api.n8n.io/api/templates/workflows";

/**
 * Server action that fetches a workflow JSON from a URL.
 * Runs server-side so there are no CORS restrictions.
 *
 * Currently has special handling for n8n template URLs
 * (extracts the workflow object from the n8n API response).
 * For all other URLs it fetches the raw JSON as-is.
 */
export async function fetchWorkflowFromUrl(url: string): Promise<string> {
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
