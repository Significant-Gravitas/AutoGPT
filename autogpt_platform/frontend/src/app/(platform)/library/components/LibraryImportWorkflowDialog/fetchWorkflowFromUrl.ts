"use server";

const N8N_URL_RE = /n8n\.io\/workflows\/(\d+)/i;
const N8N_TEMPLATES_API = "https://api.n8n.io/api/templates/workflows";

/** Hostnames allowed for URL-based workflow import (SSRF prevention). */
const ALLOWED_HOSTS = ["n8n.io", "api.n8n.io"];

/** Max response body size (10 MB) to prevent memory exhaustion. */
const MAX_RESPONSE_BYTES = 10 * 1024 * 1024;

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

/**
 * Fetch a URL and return the response text, rejecting redirects to
 * non-allowed hosts and enforcing a response size limit.
 */
async function safeFetch(url: string): Promise<Response> {
  const res = await fetch(url, { redirect: "follow" });

  // After following redirects, verify the final URL is still on an allowed host
  if (res.url) {
    const finalHost = new URL(res.url).hostname;
    if (
      !ALLOWED_HOSTS.some((h) => finalHost === h || finalHost.endsWith(`.${h}`))
    ) {
      throw new Error("URL redirected to a disallowed host.");
    }
  }

  // Reject responses that are too large
  const contentLength = res.headers.get("content-length");
  if (contentLength && parseInt(contentLength, 10) > MAX_RESPONSE_BYTES) {
    throw new Error("Response too large.");
  }

  return res;
}

async function fetchGenericJson(url: string): Promise<string> {
  const res = await safeFetch(url);
  if (!res.ok) throw new Error(`Failed to fetch workflow (${res.status})`);
  const text = await res.text();
  if (text.length > MAX_RESPONSE_BYTES) throw new Error("Response too large.");
  JSON.parse(text); // validate
  return text;
}

async function fetchN8nWorkflow(templateId: string): Promise<string> {
  const res = await safeFetch(`${N8N_TEMPLATES_API}/${templateId}`);
  if (!res.ok) throw new Error(`n8n template not found (${res.status})`);

  const text = await res.text();
  if (text.length > MAX_RESPONSE_BYTES) throw new Error("Response too large.");
  const data = JSON.parse(text);
  const template = data?.workflow ?? data;
  const workflow = template?.workflow ?? template;
  if (!workflow?.nodes) throw new Error("Unexpected n8n API response format");
  if (!workflow.name) workflow.name = template?.name ?? data?.name ?? "";
  return JSON.stringify(workflow);
}
