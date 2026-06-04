import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { LocalPCBadge } from "../LocalPCBadge";

vi.mock("@sentry/nextjs", () => ({
  captureException: vi.fn(),
}));

afterEach(cleanup);

describe("LocalPCBadge", () => {
  it("renders the pill label", () => {
    render(<LocalPCBadge />);
    expect(screen.getByText(/Local PC mode/i)).toBeInTheDocument();
  });
});
