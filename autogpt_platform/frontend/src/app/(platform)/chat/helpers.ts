/**
 * Maps internal tool names to user-friendly display names with emojis.
 *
 * @param toolName - The internal tool name from the backend
 * @returns A user-friendly display name with an emoji prefix
 */
export function getToolDisplayName(toolName: string): string {
  const toolDisplayNames: Record<string, string> = {
    find_agent: "🔍 Search Marketplace",
    get_agent_details: "📋 Get Agent Details",
    check_credentials: "🔑 Check Credentials",
    setup_agent: "⚙️ Setup Agent",
    run_agent: "▶️ Run Agent",
    get_required_setup_info: "📝 Get Setup Requirements",
  };
  return toolDisplayNames[toolName] || toolName;
}

/** Validate UUID v4 format */
export function isValidUUID(value: string): boolean {
  const uuidRegex =
    /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  return uuidRegex.test(value);
}
