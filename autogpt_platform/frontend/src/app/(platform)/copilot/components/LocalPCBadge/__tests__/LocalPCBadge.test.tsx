import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { LocalPCBadge } from "../LocalPCBadge";

vi.mock("@sentry/nextjs", () => ({
  captureException: vi.fn(),
}));

vi.mock("nuqs", () => ({
  parseAsString: { parse: (v: string | null) => v },
  useQueryState: () => ["test-session-id", vi.fn()],
}));

const mockExecutor = vi.fn();
vi.mock("../../../hooks/useLocalPCExecutor", () => ({
  useLocalPCExecutor: () => mockExecutor(),
}));

afterEach(() => {
  cleanup();
  mockExecutor.mockReset();
});

describe("LocalPCBadge", () => {
  it("falls back to the static 'Local PC mode' label when no shim is connected", () => {
    mockExecutor.mockReturnValue({ data: { kind: "none" } });
    render(<LocalPCBadge />);
    expect(screen.getByText("Local PC mode")).toBeInTheDocument();
  });

  it("renders the user's machine when a shim is connected", () => {
    mockExecutor.mockReturnValue({
      data: {
        kind: "shim",
        platform: "darwin",
        arch: "arm64",
        allowed_root: "/Users/test/autogpt-workspace",
        machine_id: "abcdef1234567890",
        shim_version: "0.1.0",
        capabilities: ["shell", "files", "computer_use"],
        computer_use_features: ["screenshot", "input"],
      },
    });
    render(<LocalPCBadge />);
    expect(screen.getByText(/Local PC: macOS arm64/i)).toBeInTheDocument();
  });

  it("renders WSL2 with the disambiguating label", () => {
    mockExecutor.mockReturnValue({
      data: {
        kind: "shim",
        platform: "wsl2",
        arch: "x86_64",
        allowed_root: "/home/test/autogpt-workspace",
        machine_id: null,
        shim_version: null,
        capabilities: null,
        computer_use_features: null,
      },
    });
    render(<LocalPCBadge />);
    expect(screen.getByText(/Windows \(WSL2\)/)).toBeInTheDocument();
  });

  it("renders the generic label while the executor query is loading", () => {
    mockExecutor.mockReturnValue({ data: undefined });
    render(<LocalPCBadge />);
    expect(screen.getByText("Local PC mode")).toBeInTheDocument();
  });
});
