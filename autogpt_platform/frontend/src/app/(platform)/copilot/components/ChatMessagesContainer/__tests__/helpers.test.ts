import { describe, expect, it } from "vitest";
import {
  WORKSPACE_FILE_PATTERN,
  buildRenderSegments,
  extractWorkspaceArtifacts,
  filePartToArtifactRef,
  isCompletedToolPart,
  isInteractiveToolPart,
  isReasoningToolPart,
  parseSpecialMarkers,
  resolveWorkspaceUrls,
  splitReasoningAndResponse,
} from "../helpers";
import type { MessagePart } from "../helpers";
import type { FileUIPart } from "ai";

function textPart(text: string): MessagePart {
  return { type: "text", text } as MessagePart;
}

function reasoningPart(text: string): MessagePart {
  return { type: "reasoning", text, state: "done" } as MessagePart;
}

function toolPart(
  toolName: string,
  state: string = "output-available",
  output: unknown = "{}",
): MessagePart {
  return {
    type: `tool-${toolName}`,
    state,
    toolCallId: `call-${toolName}`,
    toolName,
    args: {},
    output,
  } as unknown as MessagePart;
}

function interactiveToolPart(
  toolName: string,
  responseType: string,
): MessagePart {
  return {
    type: `tool-${toolName}`,
    state: "output-available",
    toolCallId: `call-${toolName}`,
    toolName,
    args: {},
    output: { type: responseType },
  } as unknown as MessagePart;
}

describe("isCompletedToolPart", () => {
  it("returns true for output-available tool part", () => {
    const part = toolPart("some_tool", "output-available");
    expect(isCompletedToolPart(part)).toBe(true);
  });

  it("returns true for output-error tool part", () => {
    const part = toolPart("some_tool", "output-error");
    expect(isCompletedToolPart(part)).toBe(true);
  });

  it("returns false for input-streaming tool part", () => {
    const part = toolPart("some_tool", "input-streaming");
    expect(isCompletedToolPart(part)).toBe(false);
  });

  it("returns false for text part", () => {
    const part = textPart("hello");
    expect(isCompletedToolPart(part)).toBe(false);
  });
});

describe("isInteractiveToolPart", () => {
  it("returns true for task_decomposition type", () => {
    const part = toolPart("decompose_goal", "output-available", {
      type: "task_decomposition",
      message: "Plan",
      goal: "Build agent",
      steps: [],
      step_count: 0,
    });
    expect(isInteractiveToolPart(part)).toBe(true);
  });

  it("returns true for setup_requirements type", () => {
    const part = toolPart("run_mcp_tool", "output-available", {
      type: "setup_requirements",
      message: "Setup needed",
    });
    expect(isInteractiveToolPart(part)).toBe(true);
  });

  it("returns true for agent_details type", () => {
    const part = toolPart("find_agent", "output-available", {
      type: "agent_details",
    });
    expect(isInteractiveToolPart(part)).toBe(true);
  });

  it("returns false for non-interactive output type", () => {
    const part = toolPart("some_tool", "output-available", {
      type: "generic_output",
    });
    expect(isInteractiveToolPart(part)).toBe(false);
  });

  it("returns false when state is not output-available", () => {
    const part = toolPart("decompose_goal", "input-streaming", {
      type: "task_decomposition",
    });
    expect(isInteractiveToolPart(part)).toBe(false);
  });

  it("returns false for non-tool parts", () => {
    const part = textPart("hello");
    expect(isInteractiveToolPart(part)).toBe(false);
  });

  it("returns false when output is null", () => {
    const part = toolPart("decompose_goal", "output-available", null);
    expect(isInteractiveToolPart(part)).toBe(false);
  });

  it("handles JSON-encoded string output", () => {
    const part = toolPart(
      "decompose_goal",
      "output-available",
      JSON.stringify({ type: "task_decomposition" }),
    );
    expect(isInteractiveToolPart(part)).toBe(true);
  });

  it("returns false for invalid JSON string output", () => {
    const part = toolPart(
      "decompose_goal",
      "output-available",
      "not valid json",
    );
    expect(isInteractiveToolPart(part)).toBe(false);
  });
});

describe("buildRenderSegments", () => {
  it("returns individual segments for custom tool types", () => {
    const parts = [
      toolPart("decompose_goal", "output-available", {
        type: "task_decomposition",
      }),
    ];
    const segments = buildRenderSegments(parts);
    expect(segments).toHaveLength(1);
    expect(segments[0].kind).toBe("part");
  });

  it("collapses consecutive generic completed tool parts", () => {
    const parts = [
      toolPart("unknown_tool_a", "output-available"),
      toolPart("unknown_tool_b", "output-available"),
    ];
    const segments = buildRenderSegments(parts);
    expect(segments).toHaveLength(1);
    expect(segments[0].kind).toBe("collapsed-group");
    if (segments[0].kind === "collapsed-group") {
      expect(segments[0].parts).toHaveLength(2);
    }
  });

  it("does not collapse custom tool types into groups", () => {
    const parts = [
      toolPart("decompose_goal", "output-available", {
        type: "task_decomposition",
      }),
      toolPart("create_agent", "output-available"),
    ];
    const segments = buildRenderSegments(parts);
    expect(segments).toHaveLength(2);
    expect(segments[0].kind).toBe("part");
    expect(segments[1].kind).toBe("part");
  });

  it("renders text parts individually", () => {
    const parts = [textPart("Hello"), textPart("World")];
    const segments = buildRenderSegments(parts);
    expect(segments).toHaveLength(2);
    expect(segments.every((s) => s.kind === "part")).toBe(true);
  });

  it("handles mixed custom tools, generic tools, and text", () => {
    const parts = [
      textPart("Plan:"),
      toolPart("decompose_goal", "output-available"),
      toolPart("generic_a", "output-available"),
      toolPart("generic_b", "output-available"),
      textPart("Done"),
    ];
    const segments = buildRenderSegments(parts);

    expect(segments[0].kind).toBe("part");
    expect(segments[1].kind).toBe("part");
    expect(segments[2].kind).toBe("collapsed-group");
    expect(segments[3].kind).toBe("part");
  });

  it("does not collapse a single generic tool part", () => {
    const parts = [toolPart("generic_a", "output-available")];
    const segments = buildRenderSegments(parts);
    expect(segments).toHaveLength(1);
    expect(segments[0].kind).toBe("part");
  });

  it("preserves baseIndex offset in part segments", () => {
    const parts = [textPart("Hello")];
    const segments = buildRenderSegments(parts, 5);
    expect(segments).toHaveLength(1);
    if (segments[0].kind === "part") {
      expect(segments[0].index).toBe(5);
    }
  });
});

describe("parseSpecialMarkers", () => {
  it("returns null marker for plain text", () => {
    const result = parseSpecialMarkers("Hello world");
    expect(result.markerType).toBeNull();
    expect(result.cleanText).toBe("Hello world");
  });

  it("detects error marker", () => {
    const result = parseSpecialMarkers(
      "Some preamble [__COPILOT_ERROR_f7a1__] Something went wrong",
    );
    expect(result.markerType).toBe("error");
    expect(result.markerText).toBe("Something went wrong");
  });

  it("detects retryable error marker", () => {
    const result = parseSpecialMarkers(
      "[__COPILOT_RETRYABLE_ERROR_a9c2__] Timeout reached",
    );
    expect(result.markerType).toBe("retryable_error");
    expect(result.markerText).toBe("Timeout reached");
  });

  it("detects system marker", () => {
    const result = parseSpecialMarkers(
      "[__COPILOT_SYSTEM_e3b0__] Session expired",
    );
    expect(result.markerType).toBe("system");
    expect(result.markerText).toBe("Session expired");
  });

  it("retryable takes precedence over regular error when both present", () => {
    const text =
      "[__COPILOT_RETRYABLE_ERROR_a9c2__] Retryable issue [__COPILOT_ERROR_f7a1__] Also error";
    const result = parseSpecialMarkers(text);
    expect(result.markerType).toBe("retryable_error");
  });

  it("strips marker from cleanText", () => {
    const result = parseSpecialMarkers(
      "Preamble text [__COPILOT_SYSTEM_e3b0__] System message",
    );
    expect(result.cleanText).toBe("Preamble text");
  });
});

describe("extractWorkspaceArtifacts", () => {
  it("extracts a single workspace:// link with its markdown title", () => {
    const text =
      "See [the report](workspace://550e8400-e29b-41d4-a716-446655440000) for details.";
    const out = extractWorkspaceArtifacts(text);
    expect(out).toHaveLength(1);
    expect(out[0].id).toBe("550e8400-e29b-41d4-a716-446655440000");
    expect(out[0].title).toBe("the report");
    expect(out[0].origin).toBe("agent");
  });

  it("falls back to a synthetic title when the URI isn't wrapped in link markdown", () => {
    const text = "raw workspace://abc12345-0000-0000-0000-000000000000 link";
    const out = extractWorkspaceArtifacts(text);
    expect(out).toHaveLength(1);
    expect(out[0].title).toBe("File abc12345");
  });

  it("skips URIs inside image markdown so images don't double-render", () => {
    const text =
      "![chart](workspace://abc12345-0000-0000-0000-000000000000#image/png)";
    expect(extractWorkspaceArtifacts(text)).toEqual([]);
  });

  it("still extracts non-image links when image links are also present", () => {
    const text =
      "![chart](workspace://aaaaaaaa-0000-0000-0000-000000000000#image/png) " +
      "and [doc](workspace://bbbbbbbb-0000-0000-0000-000000000000)";
    const out = extractWorkspaceArtifacts(text);
    expect(out).toHaveLength(1);
    expect(out[0].id).toBe("bbbbbbbb-0000-0000-0000-000000000000");
  });

  it("deduplicates repeated references to the same artifact id", () => {
    const text =
      "[A](workspace://11111111-0000-0000-0000-000000000000) and " +
      "[A again](workspace://11111111-0000-0000-0000-000000000000)";
    const out = extractWorkspaceArtifacts(text);
    expect(out).toHaveLength(1);
  });

  it("returns empty when no workspace URIs are present", () => {
    expect(extractWorkspaceArtifacts("plain text, no links")).toEqual([]);
  });

  it("picks up the mime hint from the URI fragment", () => {
    const text =
      "![v](workspace://cccccccc-0000-0000-0000-000000000000#video/mp4) " +
      "[d](workspace://dddddddd-0000-0000-0000-000000000000#application/pdf)";
    const out = extractWorkspaceArtifacts(text);
    expect(out).toHaveLength(1);
    expect(out[0].mimeType).toBe("application/pdf");
  });
});

describe("filePartToArtifactRef", () => {
  it("returns null without a url", () => {
    expect(
      filePartToArtifactRef({ type: "file", url: "", filename: "x" } as any),
    ).toBeNull();
  });

  it("returns null for URLs that don't match the workspace file pattern", () => {
    expect(
      filePartToArtifactRef({
        type: "file",
        url: "https://example.com/file.txt",
        filename: "file.txt",
      } as any),
    ).toBeNull();
  });

  it("extracts id from the workspace proxy URL", () => {
    const ref = filePartToArtifactRef({
      type: "file",
      url: "/api/proxy/api/workspace/files/550e8400-e29b-41d4-a716-446655440000/download",
      filename: "report.pdf",
      mediaType: "application/pdf",
    } as any);
    expect(ref?.id).toBe("550e8400-e29b-41d4-a716-446655440000");
    expect(ref?.title).toBe("report.pdf");
    expect(ref?.mimeType).toBe("application/pdf");
  });

  it("defaults origin to user-upload but accepts an override", () => {
    const url =
      "/api/proxy/api/workspace/files/550e8400-e29b-41d4-a716-446655440000/download";
    const defaulted = filePartToArtifactRef({
      type: "file",
      url,
      filename: "a.txt",
    } as any);
    expect(defaulted?.origin).toBe("user-upload");
    const overridden = filePartToArtifactRef(
      { type: "file", url, filename: "a.txt" } as any,
      "agent",
    );
    expect(overridden?.origin).toBe("agent");
  });
});

describe("isReasoningToolPart", () => {
  it("returns true for reasoning/search tools", () => {
    const reasoningTools = [
      "find_block",
      "find_agent",
      "find_library_agent",
      "search_docs",
      "get_doc_page",
      "search_feature_requests",
      "ask_question",
    ];
    for (const name of reasoningTools) {
      expect(isReasoningToolPart(toolPart(name))).toBe(true);
    }
  });

  it("returns false for action tools", () => {
    const actionTools = [
      "run_block",
      "run_agent",
      "create_agent",
      "edit_agent",
      "run_mcp_tool",
      "schedule_agent",
      "continue_run_block",
    ];
    for (const name of actionTools) {
      expect(isReasoningToolPart(toolPart(name))).toBe(false);
    }
  });

  it("returns false for text parts", () => {
    expect(isReasoningToolPart(textPart("hello"))).toBe(false);
  });
});

describe("splitReasoningAndResponse", () => {
  it("returns all parts as response when there are no tools", () => {
    const parts = [textPart("Hello"), textPart("World")];
    const result = splitReasoningAndResponse(parts);
    expect(result.reasoning).toEqual([]);
    expect(result.response).toEqual(parts);
  });

  it("splits on reasoning tools — text before goes to reasoning", () => {
    const parts = [
      textPart("Let me search..."),
      toolPart("find_block"),
      textPart("Here is your answer"),
    ];
    const result = splitReasoningAndResponse(parts);
    expect(result.reasoning).toHaveLength(2);
    expect(result.response).toHaveLength(1);
    expect((result.response[0] as { text: string }).text).toBe(
      "Here is your answer",
    );
  });

  it("does NOT split on action tools — response before run_block stays visible", () => {
    const parts = [
      textPart("Here is my answer"),
      toolPart("run_block"),
      textPart("Block finished"),
    ];
    const result = splitReasoningAndResponse(parts);
    expect(result.reasoning).toEqual([]);
    expect(result.response).toEqual(parts);
  });

  it("splits only on reasoning tools when both reasoning and action tools are present", () => {
    const parts = [
      textPart("Planning..."),
      toolPart("search_docs"),
      textPart("Found it. Running now."),
      toolPart("run_block"),
      textPart("Done!"),
    ];
    const result = splitReasoningAndResponse(parts);
    expect(result.reasoning).toHaveLength(2);
    expect(result.response).toHaveLength(3);
    expect((result.response[0] as { text: string }).text).toBe(
      "Found it. Running now.",
    );
  });

  it("returns all as response when reasoning tools have no text after them", () => {
    const parts = [
      textPart("Hello"),
      toolPart("find_agent"),
      toolPart("run_block"),
    ];
    const result = splitReasoningAndResponse(parts);
    expect(result.reasoning).toEqual([]);
    expect(result.response).toEqual(parts);
  });

  it("handles multiple reasoning tools correctly", () => {
    const parts = [
      textPart("Searching..."),
      toolPart("find_block"),
      textPart("Found one, searching more..."),
      toolPart("search_docs"),
      textPart("Here are the results"),
    ];
    const result = splitReasoningAndResponse(parts);
    expect(result.reasoning).toHaveLength(4);
    expect(result.response).toHaveLength(1);
    expect((result.response[0] as { text: string }).text).toBe(
      "Here are the results",
    );
  });

  it("handles action tool after response text without hiding the response", () => {
    const parts = [
      toolPart("find_block"),
      textPart("I found it! Let me run it."),
      toolPart("run_agent"),
    ];
    const result = splitReasoningAndResponse(parts);
    expect(result.reasoning).toHaveLength(1);
    expect(result.response).toHaveLength(2);
    expect((result.response[0] as { text: string }).text).toBe(
      "I found it! Let me run it.",
    );
  });

  it("returns empty arrays for an empty parts list", () => {
    const result = splitReasoningAndResponse([]);
    expect(result.reasoning).toEqual([]);
    expect(result.response).toEqual([]);
  });

  it("pins interactive reasoning tools into the response (object output)", () => {
    const askQuestion = interactiveToolPart(
      "ask_question",
      "input_validation_error",
    );
    const parts = [
      textPart("Let me check..."),
      askQuestion,
      textPart("Here's the result"),
    ];
    const result = splitReasoningAndResponse(parts);
    // Non-interactive reasoning (the text) stays in reasoning; the interactive
    // tool is pinned to the front of the response so it remains visible.
    expect(result.reasoning).toEqual([parts[0]]);
    expect(result.response).toHaveLength(2);
    expect(result.response[0]).toBe(askQuestion);
    expect((result.response[1] as { text: string }).text).toBe(
      "Here's the result",
    );
  });

  it("pins interactive reasoning tools even when output is a JSON string", () => {
    const askQuestion = {
      type: "tool-ask_question",
      state: "output-available",
      toolCallId: "call-ask_question",
      toolName: "ask_question",
      args: {},
      output: JSON.stringify({ type: "need_login" }),
    } as unknown as MessagePart;
    const parts = [
      toolPart("find_block"),
      askQuestion,
      textPart("Please log in and try again"),
    ];
    const result = splitReasoningAndResponse(parts);
    expect(result.reasoning).toEqual([parts[0]]);
    expect(result.response).toHaveLength(2);
    expect(result.response[0]).toBe(askQuestion);
  });

  it("keeps non-interactive reasoning tools in reasoning", () => {
    const parts = [
      toolPart("find_block"),
      toolPart("search_docs"),
      textPart("Answer"),
    ];
    const result = splitReasoningAndResponse(parts);
    expect(result.reasoning).toHaveLength(2);
    expect(result.reasoning[0]).toBe(parts[0]);
    expect(result.reasoning[1]).toBe(parts[1]);
    expect(result.response).toHaveLength(1);
  });

  it("moves a native reasoning part into reasoning when text follows", () => {
    const parts = [reasoningPart("Thinking through this..."), textPart("Done")];
    const result = splitReasoningAndResponse(parts);
    expect(result.reasoning).toHaveLength(1);
    expect(result.reasoning[0]).toBe(parts[0]);
    expect(result.response).toHaveLength(1);
    expect((result.response[0] as { text: string }).text).toBe("Done");
  });

  it("keeps a trailing native reasoning part in response when no text follows", () => {
    const parts = [textPart("Hello"), reasoningPart("Thinking...")];
    const result = splitReasoningAndResponse(parts);
    expect(result.reasoning).toEqual([]);
    expect(result.response).toEqual(parts);
  });

  it("sweeps a native reasoning part emitted after the last reasoning tool into reasoning", () => {
    const parts = [
      toolPart("find_block"),
      reasoningPart("Post-tool thinking"),
      textPart("Final answer"),
    ];
    const result = splitReasoningAndResponse(parts);
    expect(result.reasoning).toHaveLength(2);
    expect(result.reasoning[0]).toBe(parts[0]);
    expect(result.reasoning[1]).toBe(parts[1]);
    expect(result.response).toHaveLength(1);
    expect((result.response[0] as { text: string }).text).toBe("Final answer");
  });

  it("splits on reasoning parts alone when no reasoning tools are present", () => {
    const parts = [
      reasoningPart("Step 1"),
      reasoningPart("Step 2"),
      textPart("Here's the answer"),
    ];
    const result = splitReasoningAndResponse(parts);
    expect(result.reasoning).toHaveLength(2);
    expect(result.response).toHaveLength(1);
    expect((result.response[0] as { text: string }).text).toBe(
      "Here's the answer",
    );
  });

  it("keeps decompose_goal output pinned to response (interactive)", () => {
    const parts = [
      textPart("Thinking..."),
      toolPart("decompose_goal", "output-available", {
        type: "task_decomposition",
      }),
    ];
    const { reasoning, response } = splitReasoningAndResponse(parts);
    expect(reasoning).toHaveLength(0);
    expect(response).toHaveLength(2);
  });

  it("keeps non-interactive tool parts that emit a block_list payload in reasoning", () => {
    const genericTool = toolPart("find_block", "output-available", {
      type: "block_list",
    });
    const parts = [
      textPart("Looking for blocks..."),
      genericTool,
      textPart("Found them."),
    ];
    const { reasoning, response } = splitReasoningAndResponse(parts);
    expect(reasoning).toHaveLength(2);
    expect(reasoning[1]).toBe(genericTool);
    expect(response).toHaveLength(1);
  });
});

// ----- Custom fileUrlBuilder threading -----------------------------------
// The public-share viewer threads a token-aware URL builder through
// these helpers so anonymous readers can render file references that
// hit the public allowlist-gated download endpoint instead of the
// auth'd workspace one.  These tests pin the contract.

describe("extractWorkspaceArtifacts with custom fileUrlBuilder", () => {
  const FILE_ID = "550e8400-e29b-41d4-a716-446655440000";

  it("routes sourceUrl through the supplied builder", () => {
    const text = `See [report](workspace://${FILE_ID}) for details.`;
    const builder = (id: string) => `/share/files/${id}.dl`;
    const out = extractWorkspaceArtifacts(text, builder);
    expect(out).toHaveLength(1);
    expect(out[0].sourceUrl).toBe(`/share/files/${FILE_ID}.dl`);
  });

  it("default builder produces the workspace-file URL", () => {
    const text = `[report](workspace://${FILE_ID})`;
    const out = extractWorkspaceArtifacts(text);
    expect(out[0].sourceUrl).toContain(`/files/${FILE_ID}/download`);
  });
});

describe("resolveWorkspaceUrls with custom fileUrlBuilder", () => {
  const FILE_ID = "550e8400-e29b-41d4-a716-446655440000";

  it("rewrites image syntax using the supplied builder", () => {
    const text = `![pic](workspace://${FILE_ID}#image/png)`;
    const builder = (id: string) => `/share/files/${id}.png`;
    const out = resolveWorkspaceUrls(text, builder);
    expect(out).toBe(`![pic](/share/files/${FILE_ID}.png)`);
  });

  it("rewrites link syntax to absolute URL with origin prefix", () => {
    const text = `Open [the file](workspace://${FILE_ID}) here.`;
    const builder = (id: string) => `/share/files/${id}.dl`;
    const out = resolveWorkspaceUrls(text, builder);
    // jsdom's window.location.origin is "http://localhost:3000".
    expect(out).toContain(`(http://localhost:3000/share/files/${FILE_ID}.dl)`);
  });

  it("default builder rewrites workspace:// to the workspace endpoint", () => {
    const text = `![pic](workspace://${FILE_ID})`;
    const out = resolveWorkspaceUrls(text);
    expect(out).toMatch(/api\/workspace\/files\/.*\/download/);
  });

  it("video MIME hint produces video: alt prefix", () => {
    const text = `![demo](workspace://${FILE_ID}#video/mp4)`;
    const builder = (id: string) => `/share/files/${id}.mp4`;
    const out = resolveWorkspaceUrls(text, builder);
    expect(out).toBe(`![video:demo](/share/files/${FILE_ID}.mp4)`);
  });
});

describe("filePartToArtifactRef with custom pattern", () => {
  const FILE_ID = "550e8400-e29b-41d4-a716-446655440000";
  const file: FileUIPart = {
    type: "file",
    filename: "report.png",
    mediaType: "image/png",
    url: `/api/proxy/api/public/shared/chats/some-token/files/${FILE_ID}/download`,
  };

  it("default pattern (workspace-file) rejects public-share URL", () => {
    expect(filePartToArtifactRef(file)).toBeNull();
  });

  it("custom pattern matching the public-share URL extracts the file ID", () => {
    const pattern =
      /\/api\/proxy\/api\/public\/shared\/chats\/[^/]+\/files\/([a-f0-9-]+)\/download/;
    const ref = filePartToArtifactRef(file, "agent", pattern);
    expect(ref?.id).toBe(FILE_ID);
    expect(ref?.title).toBe("report.png");
    expect(ref?.mimeType).toBe("image/png");
  });

  it("returns null when url has no file", () => {
    expect(
      filePartToArtifactRef({ ...file, url: "" } as FileUIPart),
    ).toBeNull();
  });

  it("WORKSPACE_FILE_PATTERN matches a workspace-file URL", () => {
    const url = `/api/proxy/api/workspace/files/${FILE_ID}/download`;
    expect(url.match(WORKSPACE_FILE_PATTERN)?.[1]).toBe(FILE_ID);
  });
});
