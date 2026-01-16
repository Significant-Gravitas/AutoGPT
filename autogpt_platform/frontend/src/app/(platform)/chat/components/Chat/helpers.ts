/**
 * Maps internal tool names to user-friendly display names with emojis.
 * @deprecated Use getToolActionPhrase or getToolCompletionPhrase for status messages
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

/**
 * Maps internal tool names to human-friendly action phrases (present continuous).
 * Used for tool call messages to indicate what action is currently happening.
 *
 * @param toolName - The internal tool name from the backend
 * @returns A human-friendly action phrase in present continuous tense
 */
export function getToolActionPhrase(toolName: string): string {
  const toolActionPhrases: Record<string, string> = {
    find_agent: "Looking for agents in the marketplace",
    agent_carousel: "Looking for agents in the marketplace",
    get_agent_details: "Learning about the agent",
    check_credentials: "Checking your credentials",
    setup_agent: "Setting up the agent",
    execution_started: "Running the agent",
    run_agent: "Running the agent",
    get_required_setup_info: "Getting setup requirements",
    schedule_agent: "Scheduling the agent to run",
  };

  // Return mapped phrase or generate human-friendly fallback
  return toolActionPhrases[toolName] || toolName;
}

/**
 * Maps internal tool names to human-friendly completion phrases (past tense).
 * Used for tool response messages to indicate what action was completed.
 *
 * @param toolName - The internal tool name from the backend
 * @returns A human-friendly completion phrase in past tense
 */
export function getToolCompletionPhrase(toolName: string): string {
  const toolCompletionPhrases: Record<string, string> = {
    find_agent: "Finished searching the marketplace",
    get_agent_details: "Got agent details",
    check_credentials: "Checked credentials",
    setup_agent: "Agent setup complete",
    run_agent: "Agent execution started",
    get_required_setup_info: "Got setup requirements",
  };

  // Return mapped phrase or generate human-friendly fallback
  return (
    toolCompletionPhrases[toolName] ||
    `Finished ${toolName.replace(/_/g, " ").replace("...", "")}`
  );
}
