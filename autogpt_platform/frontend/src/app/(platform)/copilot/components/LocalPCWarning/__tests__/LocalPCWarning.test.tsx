import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { LocalPCWarning } from "../LocalPCWarning";
import { Key, storage } from "@/services/storage/local-storage";

vi.mock("@sentry/nextjs", () => ({
  captureException: vi.fn(),
}));

vi.mock("@/services/environment", () => ({
  environment: {
    isServerSide: vi.fn(() => false),
    isClientSide: vi.fn(() => true),
    getAGPTServerApiUrl: vi.fn(() => "http://localhost:8006/api"),
  },
}));

beforeEach(() => {
  storage.remove(Key.COPILOT_LOCAL_PC_WARNING_ACKED);
});

afterEach(() => {
  cleanup();
  storage.remove(Key.COPILOT_LOCAL_PC_WARNING_ACKED);
});

describe("LocalPCWarning", () => {
  it("shows the modal on first activation", async () => {
    render(<LocalPCWarning />);
    expect(
      await screen.findByText(/Experimental: code runs on your real machine/i),
    ).toBeInTheDocument();
  });

  it("hides itself after acknowledgement and writes the flag", async () => {
    const user = userEvent.setup();
    render(<LocalPCWarning />);
    const ack = await screen.findByRole("button", {
      name: /I understand — continue/i,
    });
    await user.click(ack);
    expect(
      screen.queryByText(/Experimental: code runs on your real machine/i),
    ).not.toBeInTheDocument();
    expect(storage.get(Key.COPILOT_LOCAL_PC_WARNING_ACKED)).toBe("true");
  });

  it("does not render when previously acknowledged", () => {
    storage.set(Key.COPILOT_LOCAL_PC_WARNING_ACKED, "true");
    render(<LocalPCWarning />);
    expect(
      screen.queryByText(/Experimental: code runs on your real machine/i),
    ).not.toBeInTheDocument();
  });

  it("explains the workspace jail and points at the audit log CLI", async () => {
    render(<LocalPCWarning />);
    expect(
      await screen.findByText(/~\/autogpt-workspace/),
    ).toBeInTheDocument();
    expect(screen.getByText(/autogpt-shim audit tail/)).toBeInTheDocument();
  });
});
