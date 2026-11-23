/** MCP catalog provider slugs whose icon filename differs from `<name without mcp_>`. */
const MCP_PROVIDER_ICONS: Record<string, string> = {
  mcp_agentmail: "agent_mail",
  mcp_aimlapi: "aiml_api",
  mcp_algolia_productivity: "algolia",
  mcp_apollo_io: "apollo",
  mcp_atlassian_forge: "atlassian",
  mcp_aws_databases: "aws",
  mcp_customer_io_eu: "customer_io",
  mcp_gojiberry_ai: "gojiberry",
  mcp_intercom_eu: "intercom",
};

/** Map a provider slug to its logo in `public/integrations`.
 *
 * The slug is scrubbed rather than trusted: provider strings reach this from
 * tool payloads and credential rows, and an unscrubbed one would let `../../`
 * walk out of the integrations directory. A slug that scrubs to nothing has no
 * icon at all, which callers render as their fallback.
 */
export function integrationIconSrc(provider: string): string | null {
  const scrubbed = provider
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, "_")
    .replace(/[^a-z0-9_]/g, "");
  const slug = MCP_PROVIDER_ICONS[scrubbed] ?? scrubbed.replace(/^mcp_/, "");
  if (slug === "microsoft" || slug === "microsoft_365_copilot")
    return "/integrations/microsoft.webp";
  return slug ? `/integrations/${slug}.png` : null;
}
