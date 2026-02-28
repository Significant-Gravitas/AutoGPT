/**
 * Unit tests for RunMCPTool/helpers.tsx
 *
 * Covers: type guards, output parsing, serverHost, getAnimationText
 */

import { expect, describe, it } from "vitest";
import {
  isDiscoveryOutput,
  isMCPToolOutput,
  isSetupRequirementsOutput,
  isErrorOutput,
  getRunMCPToolOutput,
  serverHost,
  getAnimationText,
  type MCPErrorOutput,
  type RunMCPToolOutput,
} from "../helpers";
import type { MCPToolsDiscoveredResponse } from "@/app/api/__generated__/models/mCPToolsDiscoveredResponse";
import type { MCPToolOutputResponse } from "@/app/api/__generated__/models/mCPToolOutputResponse";
import type { SetupRequirementsResponse } from "@/app/api/__generated__/models/setupRequirementsResponse";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const DISCOVERY: MCPToolsDiscoveredResponse = {
  type: "mcp_tools_discovered",
  message: "",
  session_id: "s1",
  server_url: "https://mcp.example.com/mcp",
  tools: [
    {
      name: "fetch",
      description: "Fetch a URL",
      input_schema: { type: "object", properties: {}, required: [] },
    },
  ],
};

const MCP_OUTPUT: MCPToolOutputResponse = {
  type: "mcp_tool_output",
  message: "",
  session_id: "s1",
  server_url: "https://mcp.example.com/mcp",
  tool_name: "fetch",
  result: "Hello World",
  success: true,
};

const SETUP: SetupRequirementsResponse = {
  type: "setup_requirements",
  message: "Credentials required",
  session_id: "s1",
  setup_info: {
    agent_id: "mcp-server",
    agent_name: "MCP Server",
  },
};

const ERROR: MCPErrorOutput = {
  type: "error",
  message: "Something went wrong",
};

// ---------------------------------------------------------------------------
// Type guards
// ---------------------------------------------------------------------------

describe("isDiscoveryOutput", () => {
  it("returns true for mcp_tools_discovered", () => {
    expect(isDiscoveryOutput(DISCOVERY)).toBe(true);
  });

  it("returns false for other types", () => {
    expect(isDiscoveryOutput(MCP_OUTPUT as unknown as RunMCPToolOutput)).toBe(
      false,
    );
    expect(isDiscoveryOutput(ERROR)).toBe(false);
  });
});

describe("isMCPToolOutput", () => {
  it("returns true for mcp_tool_output", () => {
    expect(isMCPToolOutput(MCP_OUTPUT)).toBe(true);
  });

  it("returns false for other types", () => {
    expect(isMCPToolOutput(DISCOVERY as unknown as RunMCPToolOutput)).toBe(
      false,
    );
    expect(isMCPToolOutput(ERROR)).toBe(false);
  });
});

describe("isSetupRequirementsOutput", () => {
  it("returns true for setup_requirements type literal", () => {
    expect(
      isSetupRequirementsOutput(SETUP as unknown as RunMCPToolOutput),
    ).toBe(true);
  });

  it("returns true for structural match (setup_info present)", () => {
    const structural = { type: "unknown", setup_info: { agent_name: "X" } };
    expect(
      isSetupRequirementsOutput(structural as unknown as RunMCPToolOutput),
    ).toBe(true);
  });

  it("returns false for error output", () => {
    expect(isSetupRequirementsOutput(ERROR)).toBe(false);
  });
});

describe("isErrorOutput", () => {
  it("returns true for error type", () => {
    expect(isErrorOutput(ERROR)).toBe(true);
  });

  it("returns false for setup_requirements (has setup_info)", () => {
    expect(isErrorOutput(SETUP as unknown as RunMCPToolOutput)).toBe(false);
  });

  it("returns false for discovery output", () => {
    expect(isErrorOutput(DISCOVERY as unknown as RunMCPToolOutput)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// getRunMCPToolOutput — output parsing
// ---------------------------------------------------------------------------

describe("getRunMCPToolOutput", () => {
  it("parses a direct object output", () => {
    const part = { output: DISCOVERY };
    expect(getRunMCPToolOutput(part)).toEqual(DISCOVERY);
  });

  it("parses a JSON-encoded string output", () => {
    const part = { output: JSON.stringify(MCP_OUTPUT) };
    const result = getRunMCPToolOutput(part);
    expect(result).not.toBeNull();
    expect(isMCPToolOutput(result!)).toBe(true);
  });

  it("returns null for empty/falsy output", () => {
    expect(getRunMCPToolOutput({ output: null })).toBeNull();
    expect(getRunMCPToolOutput({ output: "" })).toBeNull();
    expect(getRunMCPToolOutput({ output: undefined })).toBeNull();
  });

  it("returns null for a plain string (not JSON)", () => {
    expect(getRunMCPToolOutput({ output: "just text" })).toBeNull();
  });

  it("returns null for an object with unknown type", () => {
    const part = { output: { type: "something_else", data: 42 } };
    expect(getRunMCPToolOutput(part)).toBeNull();
  });

  it("returns null for a non-object part", () => {
    expect(getRunMCPToolOutput(null)).toBeNull();
    expect(getRunMCPToolOutput("string")).toBeNull();
    expect(getRunMCPToolOutput(42)).toBeNull();
  });

  it("falls back to structural check for tool_name field", () => {
    const raw = {
      tool_name: "fetch",
      result: "ok",
      server_url: "https://x.com",
    };
    const part = { output: raw };
    const result = getRunMCPToolOutput(part);
    expect(result).not.toBeNull();
    expect((result as MCPToolOutputResponse).tool_name).toBe("fetch");
  });

  it("falls back to structural check for tools field", () => {
    const raw = { tools: [], server_url: "https://x.com" };
    const part = { output: raw };
    const result = getRunMCPToolOutput(part);
    expect(result).not.toBeNull();
    expect((result as MCPToolsDiscoveredResponse).tools).toEqual([]);
  });

  it("falls back to structural check for setup_info field", () => {
    const raw = { setup_info: { agent_name: "My MCP" }, credentials: [] };
    const part = { output: raw };
    const result = getRunMCPToolOutput(part);
    expect(result).not.toBeNull();
    expect(isSetupRequirementsOutput(result!)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// serverHost
// ---------------------------------------------------------------------------

describe("serverHost", () => {
  it("extracts hostname from a standard URL", () => {
    expect(serverHost("https://mcp.example.com/mcp")).toBe("mcp.example.com");
  });

  it("strips port from hostname", () => {
    expect(serverHost("https://mcp.example.com:8080/mcp")).toBe(
      "mcp.example.com",
    );
  });

  it("returns the raw string for an unparseable URL", () => {
    expect(serverHost("not-a-url")).toBe("not-a-url");
  });

  it("handles URLs with subpaths", () => {
    expect(serverHost("https://api.remote.mcp.io/v1/mcp")).toBe(
      "api.remote.mcp.io",
    );
  });
});

// ---------------------------------------------------------------------------
// getAnimationText
// ---------------------------------------------------------------------------

describe("getAnimationText", () => {
  const BASE = {
    input: {
      server_url: "https://mcp.example.com/mcp",
    },
  };

  it("shows discovery text while streaming with just server_url", () => {
    const text = getAnimationText({
      state: "input-streaming",
      ...BASE,
    });
    expect(text).toContain("Discovering");
    expect(text).toContain("mcp.example.com");
  });

  it("shows tool call text when tool_name is set", () => {
    const text = getAnimationText({
      state: "input-available",
      input: { server_url: "https://mcp.example.com/mcp", tool_name: "fetch" },
    });
    expect(text).toContain("fetch");
    expect(text).toContain("mcp.example.com");
  });

  it("includes query argument preview when tool_arguments has a query key", () => {
    const text = getAnimationText({
      state: "input-available",
      input: {
        server_url: "https://mcp.example.com/mcp",
        tool_name: "search",
        tool_arguments: { query: "my search term" },
      },
    });
    expect(text).toContain(`"my search term"`);
    expect(text).toContain("search");
  });

  it("falls back to first string value when no known query key is present", () => {
    const text = getAnimationText({
      state: "input-available",
      input: {
        server_url: "https://mcp.example.com/mcp",
        tool_name: "get_page",
        tool_arguments: { page_id: "abc123" },
      },
    });
    expect(text).toContain(`"abc123"`);
  });

  it("shows no arg preview when tool_arguments is empty", () => {
    const text = getAnimationText({
      state: "input-available",
      input: {
        server_url: "https://mcp.example.com/mcp",
        tool_name: "list_users",
        tool_arguments: {},
      },
    });
    expect(text).toBe("Calling list_users on mcp.example.com");
  });

  it("shows ran text on output-available for tool output", () => {
    const text = getAnimationText({
      state: "output-available",
      output: MCP_OUTPUT,
      ...BASE,
    });
    expect(text).toContain("Ran");
    expect(text).toContain("fetch");
  });

  it("shows discovered text on output-available for discovery output", () => {
    const text = getAnimationText({
      state: "output-available",
      output: DISCOVERY,
      input: { server_url: "https://mcp.example.com/mcp" },
    });
    expect(text).toContain("Discovered");
    expect(text).toContain("1");
  });

  it("shows setup label on output-available for setup requirements", () => {
    const text = getAnimationText({
      state: "output-available",
      output: SETUP,
      ...BASE,
    });
    expect(text).toContain("MCP Server");
  });

  it("shows generic error on output-error state", () => {
    const text = getAnimationText({ state: "output-error", ...BASE });
    expect(text.toLowerCase()).toContain("error");
  });

  it("shows fallback for unknown state", () => {
    const text = getAnimationText({
      state: "input-streaming",
      input: {},
    });
    // No server_url — still shows something
    expect(typeof text).toBe("string");
    expect(text.length).toBeGreaterThan(0);
  });
});
