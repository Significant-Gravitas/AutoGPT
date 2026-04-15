import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MCPToolOutputCard } from "../MCPToolOutputCard";

function makeOutput(overrides: Record<string, unknown> = {}) {
  return {
    type: "mcp_tool_output" as const,
    message: "MCP tool 'fetch' executed successfully.",
    server_url: "https://mcp.example.com/mcp",
    tool_name: "fetch",
    result: "Hello, world!",
    success: true,
    session_id: "test-session",
    ...overrides,
  };
}

describe("MCPToolOutputCard", () => {
  it("renders tool name and message", () => {
    render(<MCPToolOutputCard output={makeOutput()} />);
    expect(screen.getByText("fetch")).toBeDefined();
    expect(
      screen.getByText("MCP tool 'fetch' executed successfully."),
    ).toBeDefined();
  });

  it("renders plain text result", () => {
    render(<MCPToolOutputCard output={makeOutput({ result: "plain text" })} />);
    expect(screen.getByText("plain text")).toBeDefined();
  });

  it("renders JSON result in code block", () => {
    const json = { status: "ok", count: 42 };
    render(<MCPToolOutputCard output={makeOutput({ result: json })} />);
    // JSON may be split across text nodes by ContentCodeBlock, so match a key substring.
    expect(screen.getByText(/"status": "ok"/)).toBeDefined();
  });

  it("renders null as '(no result)'", () => {
    render(<MCPToolOutputCard output={makeOutput({ result: null })} />);
    expect(screen.getByText("(no result)")).toBeDefined();
  });

  it("renders image result as img element", () => {
    const imageResult = {
      type: "image",
      data: "abc123==",
      mimeType: "image/png",
    };
    render(<MCPToolOutputCard output={makeOutput({ result: imageResult })} />);
    const img = screen.getByRole("img");
    expect(img.getAttribute("src")).toBe("data:image/png;base64,abc123==");
    expect(img.getAttribute("alt")).toBe("Result from fetch");
  });
});
