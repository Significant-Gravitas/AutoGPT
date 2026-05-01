import {
  cleanup,
  render,
  screen,
  fireEvent,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";

const mockPush = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: mockPush,
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => "/copilot",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

import { RateLimitResetDialog } from "../RateLimitResetDialog";

afterEach(() => {
  cleanup();
  mockPush.mockReset();
});

describe("RateLimitResetDialog", () => {
  it("renders the dialog title and body when open", () => {
    render(
      <RateLimitResetDialog isOpen={true} onClose={vi.fn()} resetsAt={null} />,
    );

    expect(screen.getByText("Daily AutoPilot limit reached")).toBeDefined();
    expect(
      screen.getByText(/You've reached your daily usage limit/),
    ).toBeDefined();
  });

  it("shows the reset time when resetsAt is provided", () => {
    const future = new Date(Date.now() + 3 * 60 * 60 * 1000).toISOString();
    render(
      <RateLimitResetDialog
        isOpen={true}
        onClose={vi.fn()}
        resetsAt={future}
      />,
    );

    expect(screen.getByText(/Resets in/)).toBeDefined();
  });

  it("omits the reset time when resetsAt is null", () => {
    render(
      <RateLimitResetDialog isOpen={true} onClose={vi.fn()} resetsAt={null} />,
    );

    const bodyText = screen.getByText(/You've reached your daily usage limit/);
    expect(bodyText.textContent).not.toContain("Resets in");
  });

  it("renders Wait for reset and Go to billing buttons", () => {
    render(
      <RateLimitResetDialog isOpen={true} onClose={vi.fn()} resetsAt={null} />,
    );

    expect(screen.getByText("Wait for reset")).toBeDefined();
    expect(screen.getByText("Go to billing")).toBeDefined();
  });

  it("calls onClose when Wait for reset is clicked", () => {
    const onClose = vi.fn();
    render(
      <RateLimitResetDialog isOpen={true} onClose={onClose} resetsAt={null} />,
    );

    fireEvent.click(screen.getByText("Wait for reset"));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("navigates to /settings/billing when Go to billing is clicked", () => {
    const onClose = vi.fn();
    render(
      <RateLimitResetDialog isOpen={true} onClose={onClose} resetsAt={null} />,
    );

    fireEvent.click(screen.getByText("Go to billing"));
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(mockPush).toHaveBeenCalledWith("/settings/billing");
  });

  it("does not render dialog content when closed", () => {
    render(
      <RateLimitResetDialog isOpen={false} onClose={vi.fn()} resetsAt={null} />,
    );

    expect(screen.queryByText("Daily AutoPilot limit reached")).toBeNull();
  });
});
