function stripInternalReasoning(content: string): string {
  return content
    .replace(/<internal_reasoning>[\s\S]*?<\/internal_reasoning>/gi, "")
    .replace(/<thinking>[\s\S]*?<\/thinking>/gi, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export function isErrorResponse(result: unknown): boolean {
  if (typeof result === "string") {
    const lower = result.toLowerCase();
    return (
      lower.startsWith("error:") ||
      lower.includes("not found") ||
      lower.includes("does not exist") ||
      lower.includes("failed to") ||
      lower.includes("unable to")
    );
  }
  if (typeof result === "object" && result !== null) {
    const response = result as Record<string, unknown>;
    return response.type === "error" || response.error !== undefined;
  }
  return false;
}

export function getErrorMessage(result: unknown): string {
  if (typeof result === "string") {
    return stripInternalReasoning(result.replace(/^error:\s*/i, ""));
  }
  if (typeof result === "object" && result !== null) {
    const response = result as Record<string, unknown>;
    if (response.error) return stripInternalReasoning(String(response.error));
    if (response.message)
      return stripInternalReasoning(String(response.message));
  }
  return "An error occurred";
}

/**
 * Check if a value is a workspace file reference.
 * Format: workspace://{fileId} or workspace://{fileId}#{mimeType}
 */
function isWorkspaceRef(value: unknown): value is string {
  return typeof value === "string" && value.startsWith("workspace://");
}

/**
 * Extract MIME type from a workspace reference fragment.
 * e.g., "workspace://abc123#video/mp4" → "video/mp4"
 * Returns undefined if no fragment is present.
 */
function getWorkspaceMimeType(value: string): string | undefined {
  const hashIndex = value.indexOf("#");
  if (hashIndex === -1) return undefined;
  return value.slice(hashIndex + 1) || undefined;
}

/**
 * Determine the media category of a workspace ref or data URI.
 * Uses the MIME type fragment on workspace refs when available,
 * falls back to output key keyword matching for older refs without it.
 */
function getMediaCategory(
  value: string,
  outputKey?: string,
): "video" | "image" | "audio" | "unknown" {
  // Data URIs carry their own MIME type
  if (value.startsWith("data:video/")) return "video";
  if (value.startsWith("data:image/")) return "image";
  if (value.startsWith("data:audio/")) return "audio";

  // Workspace refs: prefer MIME type fragment
  if (isWorkspaceRef(value)) {
    const mime = getWorkspaceMimeType(value);
    if (mime) {
      if (mime.startsWith("video/")) return "video";
      if (mime.startsWith("image/")) return "image";
      if (mime.startsWith("audio/")) return "audio";
      return "unknown";
    }

    // Fallback: keyword matching on output key for older refs without fragment
    if (outputKey) {
      const lowerKey = outputKey.toLowerCase();

      const videoKeywords = [
        "video",
        "mp4",
        "mov",
        "avi",
        "webm",
        "movie",
        "clip",
      ];
      if (videoKeywords.some((kw) => lowerKey.includes(kw))) return "video";

      const imageKeywords = [
        "image",
        "img",
        "photo",
        "picture",
        "thumbnail",
        "avatar",
        "icon",
        "screenshot",
      ];
      if (imageKeywords.some((kw) => lowerKey.includes(kw))) return "image";
    }

    // Default to image for backward compatibility
    return "image";
  }

  return "unknown";
}

/**
 * Format a single output value, converting workspace refs to markdown images/videos.
 * Videos use a "video:" alt-text prefix so the MarkdownContent renderer can
 * distinguish them from images and render a <video> element.
 */
function formatOutputValue(value: unknown, outputKey?: string): string {
  if (typeof value === "string") {
    const category = getMediaCategory(value, outputKey);

    if (category === "video") {
      // Format with "video:" prefix so MarkdownContent renders <video>
      return `![video:${outputKey || "Video"}](${value})`;
    }

    if (category === "image") {
      return `![${outputKey || "Generated image"}](${value})`;
    }

    // For audio, unknown workspace refs, data URIs, etc. - return as-is
    return value;
  }

  if (Array.isArray(value)) {
    return value
      .map((item, idx) => formatOutputValue(item, `${outputKey}_${idx}`))
      .join("\n\n");
  }

  if (typeof value === "object" && value !== null) {
    return JSON.stringify(value, null, 2);
  }

  return String(value);
}

function getToolCompletionPhrase(toolName: string): string {
  const toolCompletionPhrases: Record<string, string> = {
    add_understanding: "Updated your business information",
    create_agent: "Created the agent",
    edit_agent: "Updated the agent",
    find_agent: "Found agents in the marketplace",
    find_block: "Found blocks",
    find_library_agent: "Found library agents",
    run_agent: "Started agent execution",
    run_block: "Executed the block",
    view_agent_output: "Retrieved agent output",
    search_docs: "Found documentation",
    get_doc_page: "Loaded documentation page",
  };

  // Return mapped phrase or generate human-friendly fallback
  return (
    toolCompletionPhrases[toolName] ||
    `Completed ${toolName.replace(/_/g, " ").replace("...", "")}`
  );
}

export function formatToolResponse(result: unknown, toolName: string): string {
  if (typeof result === "string") {
    const trimmed = result.trim();
    if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
      try {
        const parsed = JSON.parse(trimmed);
        return formatToolResponse(parsed, toolName);
      } catch {
        return stripInternalReasoning(trimmed);
      }
    }
    return stripInternalReasoning(result);
  }

  if (typeof result !== "object" || result === null) {
    return String(result);
  }

  const response = result as Record<string, unknown>;

  // Handle different response types
  const responseType = response.type as string | undefined;

  if (!responseType) {
    if (response.message) {
      return String(response.message);
    }
    return getToolCompletionPhrase(toolName);
  }

  switch (responseType) {
    case "agents_found":
      const agents = (response.agents as Array<{ name: string }>) || [];
      const count =
        typeof response.count === "number" && !isNaN(response.count)
          ? response.count
          : agents.length;
      if (count === 0) {
        return "No agents found matching your search.";
      }
      return `Found ${count} agent${count === 1 ? "" : "s"}: ${agents
        .slice(0, 3)
        .map((a) => a.name)
        .join(", ")}${count > 3 ? ` and ${count - 3} more` : ""}`;

    case "agent_details":
      const agent = response.agent as { name: string; description?: string };
      if (agent) {
        return `Agent: ${agent.name}${agent.description ? `\n\n${agent.description}` : ""}`;
      }
      break;

    case "block_list":
      const blocks = (response.blocks as Array<{ name: string }>) || [];
      const blockCount =
        typeof response.count === "number" && !isNaN(response.count)
          ? response.count
          : blocks.length;
      if (blockCount === 0) {
        return "No blocks found matching your search.";
      }
      return `Found ${blockCount} block${blockCount === 1 ? "" : "s"}: ${blocks
        .slice(0, 3)
        .map((b) => b.name)
        .join(", ")}${blockCount > 3 ? ` and ${blockCount - 3} more` : ""}`;

    case "block_output":
      const blockName = (response.block_name as string) || "Block";
      const outputs = response.outputs as Record<string, unknown[]> | undefined;
      if (outputs && Object.keys(outputs).length > 0) {
        const formattedOutputs: string[] = [];

        for (const [key, values] of Object.entries(outputs)) {
          if (!Array.isArray(values) || values.length === 0) continue;

          // Format each value in the output array
          for (const value of values) {
            const formatted = formatOutputValue(value, key);
            if (formatted) {
              formattedOutputs.push(formatted);
            }
          }
        }

        if (formattedOutputs.length > 0) {
          return `${blockName} executed successfully.\n\n${formattedOutputs.join("\n\n")}`;
        }
        return `${blockName} executed successfully.`;
      }
      return `${blockName} executed successfully.`;

    case "doc_search_results":
      const docResults = (response.results as Array<{ title: string }>) || [];
      const docCount =
        typeof response.count === "number" && !isNaN(response.count)
          ? response.count
          : docResults.length;
      if (docCount === 0) {
        return "No documentation found matching your search.";
      }
      return `Found ${docCount} documentation result${docCount === 1 ? "" : "s"}: ${docResults
        .slice(0, 3)
        .map((r) => r.title)
        .join(", ")}${docCount > 3 ? ` and ${docCount - 3} more` : ""}`;

    case "doc_page":
      const title = (response.title as string) || "Documentation";
      const content = (response.content as string) || "";
      if (content) {
        const preview = content.substring(0, 200).trim();
        return `${title}\n\n${preview}${content.length > 200 ? "..." : ""}`;
      }
      return title;

    case "understanding_updated":
      const currentUnderstanding = response.current_understanding as
        | Record<string, unknown>
        | undefined;
      const fields = (response.updated_fields as string[]) || [];

      if (response.message && typeof response.message === "string") {
        let message = response.message;
        if (currentUnderstanding) {
          for (const [key, value] of Object.entries(currentUnderstanding)) {
            if (value !== null && value !== undefined && value !== "") {
              const placeholder = key;
              const actualValue = String(value);
              message = message.replace(
                new RegExp(`\\b${placeholder}\\b`, "g"),
                actualValue,
              );
            }
          }
        }
        return message;
      }

      if (
        currentUnderstanding &&
        Object.keys(currentUnderstanding).length > 0
      ) {
        const understandingEntries = Object.entries(currentUnderstanding)
          .filter(
            ([_, value]) =>
              value !== null && value !== undefined && value !== "",
          )
          .map(([key, value]) => {
            if (key === "user_name" && typeof value === "string") {
              return `Updated information for ${value}`;
            }
            return `${key}: ${String(value)}`;
          });
        if (understandingEntries.length > 0) {
          return understandingEntries[0];
        }
      }
      if (fields.length > 0) {
        return `Updated business information: ${fields.join(", ")}`;
      }
      return "Updated your business information.";

    case "agent_saved":
      const agentName = (response.agent_name as string) || "Agent";
      return `Successfully saved "${agentName}" to your library.`;

    case "agent_preview":
      const previewAgentName = (response.agent_name as string) || "Agent";
      const nodeCount = (response.node_count as number) || 0;
      const linkCount = (response.link_count as number) || 0;
      const description = (response.description as string) || "";
      let previewText = `Preview: "${previewAgentName}"`;
      if (description) {
        previewText += `\n\n${description}`;
      }
      previewText += `\n\n${nodeCount} node${nodeCount === 1 ? "" : "s"}, ${linkCount} link${linkCount === 1 ? "" : "s"}`;
      return previewText;

    case "clarification_needed":
      const questions =
        (response.questions as Array<{ question: string }>) || [];
      if (questions.length === 0) {
        return response.message
          ? String(response.message)
          : "I need more information to proceed.";
      }
      if (questions.length === 1) {
        return questions[0].question;
      }
      return `I need clarification on ${questions.length} points:\n\n${questions
        .map((q, i) => `${i + 1}. ${q.question}`)
        .join("\n")}`;

    case "agent_output":
      if (response.message) {
        return String(response.message);
      }
      const outputAgentName = (response.agent_name as string) || "Agent";
      const execution = response.execution as
        | {
            status?: string;
            outputs?: Record<string, unknown>;
          }
        | undefined;
      if (execution) {
        const status = execution.status || "completed";
        const outputs = execution.outputs || {};
        const outputKeys = Object.keys(outputs);
        if (outputKeys.length > 0) {
          return `${outputAgentName} execution ${status}. Outputs: ${outputKeys.join(", ")}`;
        }
        return `${outputAgentName} execution ${status}.`;
      }
      return `${outputAgentName} output retrieved.`;

    case "execution_started":
      const execAgentName = (response.graph_name as string) || "Agent";
      if (response.message) {
        return String(response.message);
      }
      return `Started execution of "${execAgentName}".`;

    case "error":
      const errorMsg =
        (response.error as string) || response.message || "An error occurred";
      return `Error: ${errorMsg}`;

    case "no_results":
      const suggestions = (response.suggestions as string[]) || [];
      let noResultsText = (response.message as string) || "No results found.";
      if (suggestions.length > 0) {
        noResultsText += `\n\nSuggestions: ${suggestions.join(", ")}`;
      }
      return noResultsText;

    default:
      // Try to extract a message field
      if (response.message) {
        return String(response.message);
      }
      // Fallback: try to stringify nicely
      if (Object.keys(response).length === 0) {
        return getToolCompletionPhrase(toolName);
      }
      // If we have a response object but no recognized type, try to format it nicely
      // Don't return raw JSON - return a completion phrase instead
      return getToolCompletionPhrase(toolName);
  }

  return getToolCompletionPhrase(toolName);
}
