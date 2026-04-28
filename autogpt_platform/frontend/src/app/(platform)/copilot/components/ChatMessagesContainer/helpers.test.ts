import { describe, expect, it } from "vitest";
import {
  extractWorkspaceArtifacts,
  filePartToArtifactRef,
  isReasoningToolPart,
  splitReasoningAndResponse,
} from "./helpers";
import type { MessagePart } from "./helpers";

function textPart(text: string): MessagePart {
  return { type: "text", text } as MessagePart;
}

function reasoningPart(text: string): MessagePart {
  return { type: "reasoning", text, state: "done" } as MessagePart;
}

function toolPart(
  toolName: string,
  state: string = "output-available",
): MessagePart {
  return {
    type: `tool-${toolName}`,
    state,
    toolCallId: `call-${toolName}`,
    toolName,
    args: {},
    output: "{}",
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
});
