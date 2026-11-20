import { describe, expect, test, vi } from "vitest";
import { render, screen, within } from "@/tests/integrations/test-utils";
import { EmailNotAllowedModal } from "../EmailNotAllowedModal";

let mockIsCloud = true;
vi.mock("@/services/environment", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/environment")>();
  return {
    ...actual,
    environment: { ...actual.environment, isCloud: () => mockIsCloud },
  };
});

vi.mock("@/lib/hooks/useBreakpoint", () => ({
  useBreakpoint: () => "lg",
  isLargeScreen: () => true,
}));

describe("EmailNotAllowedModal", () => {
  test("renders dialog with accessible title when open", () => {
    render(<EmailNotAllowedModal isOpen={true} onClose={vi.fn()} />);

    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByText("Email Not Allowed")).toBeDefined();
  });

  test("renders waitlist content when open", () => {
    render(<EmailNotAllowedModal isOpen={true} onClose={vi.fn()} />);

    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByText(/closed beta/i)).toBeDefined();
    expect(within(dialog).getByText(/join waitlist/i)).toBeDefined();
  });

  test("points a self-hoster at their administrator, not the hosted waitlist", () => {
    mockIsCloud = false;
    render(<EmailNotAllowedModal isOpen={true} onClose={vi.fn()} />);
    mockIsCloud = true;

    const dialog = screen.getByRole("dialog");
    expect(
      within(dialog).getByText(/only allows approved emails/i),
    ).toBeDefined();
    expect(within(dialog).queryByText(/closed beta/i)).toBeNull();
    expect(within(dialog).queryByText(/join waitlist/i)).toBeNull();
  });

  test("does not render dialog when closed", () => {
    render(<EmailNotAllowedModal isOpen={false} onClose={vi.fn()} />);

    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
