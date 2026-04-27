import { describe, expect, test, vi, afterEach } from "vitest";
import {
  render,
  screen,
  fireEvent,
  within,
  cleanup,
} from "@/tests/integrations/test-utils";
import { SharePlatformDialog } from "../SharePlatformDialog";

const mockPush = vi.fn();

vi.mock("next/navigation", async () => {
  const actual =
    await vi.importActual<typeof import("next/navigation")>("next/navigation");
  return {
    ...actual,
    useRouter: () => ({
      push: mockPush,
      replace: vi.fn(),
      prefetch: vi.fn(),
      back: vi.fn(),
      forward: vi.fn(),
      refresh: vi.fn(),
    }),
    usePathname: () => "/",
    useSearchParams: () => new URLSearchParams(),
    useParams: () => ({}),
  };
});

describe("SharePlatformDialog", () => {
  afterEach(() => {
    cleanup();
    mockPush.mockClear();
  });

  test("renders all platform buttons when open", () => {
    render(<SharePlatformDialog open={true} onOpenChange={vi.fn()} />);

    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByText("X (Twitter)")).toBeDefined();
    expect(within(dialog).getByText("LinkedIn")).toBeDefined();
    expect(within(dialog).getByText("Reddit")).toBeDefined();
    expect(within(dialog).getByText("Other")).toBeDefined();
  });

  test("renders dialog title and description", () => {
    render(<SharePlatformDialog open={true} onOpenChange={vi.fn()} />);

    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByText("Share AutoGPT")).toBeDefined();
    expect(
      within(dialog).getByText(
        "Pick a platform and the Copilot will help you create a sharing automation.",
      ),
    ).toBeDefined();
  });

  test("clicking a platform navigates to copilot with encoded prompt", () => {
    const onOpenChange = vi.fn();
    render(<SharePlatformDialog open={true} onOpenChange={onOpenChange} />);

    const dialog = screen.getByRole("dialog");
    fireEvent.click(within(dialog).getByText("X (Twitter)"));

    expect(onOpenChange).toHaveBeenCalledWith(false);
    expect(mockPush).toHaveBeenCalledWith(
      expect.stringContaining("/copilot#prompt="),
    );
    expect(mockPush).toHaveBeenCalledWith(
      expect.stringContaining("X%2FTwitter"),
    );
  });

  test("clicking LinkedIn navigates with LinkedIn-specific prompt", () => {
    const onOpenChange = vi.fn();
    render(<SharePlatformDialog open={true} onOpenChange={onOpenChange} />);

    const dialog = screen.getByRole("dialog");
    fireEvent.click(within(dialog).getByText("LinkedIn"));

    expect(onOpenChange).toHaveBeenCalledWith(false);
    expect(mockPush).toHaveBeenCalledWith(expect.stringContaining("LinkedIn"));
  });

  test("does not render dialog content when closed", () => {
    render(<SharePlatformDialog open={false} onOpenChange={vi.fn()} />);

    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
