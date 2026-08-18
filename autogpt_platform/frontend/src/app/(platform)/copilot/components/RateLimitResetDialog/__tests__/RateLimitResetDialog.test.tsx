import {
  cleanup,
  render,
  screen,
  fireEvent,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

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

const mockWindowOpen = vi.fn();

beforeEach(() => {
  // jsdom's window.open returns null and pollutes the test output; spy and
  // suppress so we can assert the URL the dialog tries to open.
  vi.spyOn(window, "open").mockImplementation(
    mockWindowOpen as unknown as typeof window.open,
  );
});

afterEach(() => {
  cleanup();
  mockPush.mockReset();
  mockWindowOpen.mockReset();
  vi.restoreAllMocks();
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

  it("renders Wait for reset and Upgrade plan buttons by default", () => {
    render(
      <RateLimitResetDialog isOpen={true} onClose={vi.fn()} resetsAt={null} />,
    );

    expect(screen.getByText("Wait for reset")).toBeDefined();
    expect(screen.getByText("Upgrade plan")).toBeDefined();
    expect(screen.queryByText("Contact us")).toBeNull();
  });

  it("calls onClose when Wait for reset is clicked", () => {
    const onClose = vi.fn();
    render(
      <RateLimitResetDialog isOpen={true} onClose={onClose} resetsAt={null} />,
    );

    fireEvent.click(screen.getByText("Wait for reset"));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("navigates to /settings/billing when Upgrade plan is clicked (no tier)", () => {
    const onClose = vi.fn();
    render(
      <RateLimitResetDialog isOpen={true} onClose={onClose} resetsAt={null} />,
    );

    fireEvent.click(screen.getByText("Upgrade plan"));
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(mockPush).toHaveBeenCalledWith("/settings/billing");
    expect(mockWindowOpen).not.toHaveBeenCalled();
  });

  it.each(["NO_TIER", "BASIC", "PRO"] as const)(
    "shows Upgrade plan and routes to /settings/billing for tier=%s",
    (tier) => {
      const onClose = vi.fn();
      render(
        <RateLimitResetDialog
          isOpen={true}
          onClose={onClose}
          resetsAt={null}
          tier={tier}
        />,
      );

      expect(screen.getByText("Upgrade plan")).toBeDefined();
      fireEvent.click(screen.getByText("Upgrade plan"));
      expect(onClose).toHaveBeenCalledTimes(1);
      expect(mockPush).toHaveBeenCalledWith("/settings/billing");
      expect(mockWindowOpen).not.toHaveBeenCalled();
    },
  );

  it.each(["MAX", "BUSINESS", "ENTERPRISE"] as const)(
    "shows Contact us and opens mailto for top-tier plan=%s",
    (tier) => {
      const onClose = vi.fn();
      render(
        <RateLimitResetDialog
          isOpen={true}
          onClose={onClose}
          resetsAt={null}
          tier={tier}
        />,
      );

      expect(screen.getByText("Contact us")).toBeDefined();
      expect(screen.queryByText("Upgrade plan")).toBeNull();

      fireEvent.click(screen.getByText("Contact us"));
      expect(onClose).toHaveBeenCalledTimes(1);
      expect(mockWindowOpen).toHaveBeenCalledWith(
        "mailto:contact@agpt.co",
        "_blank",
        "noopener,noreferrer",
      );
      expect(mockPush).not.toHaveBeenCalled();
    },
  );

  it("does not render dialog content when closed", () => {
    render(
      <RateLimitResetDialog isOpen={false} onClose={vi.fn()} resetsAt={null} />,
    );

    expect(screen.queryByText("Daily AutoPilot limit reached")).toBeNull();
  });
});
