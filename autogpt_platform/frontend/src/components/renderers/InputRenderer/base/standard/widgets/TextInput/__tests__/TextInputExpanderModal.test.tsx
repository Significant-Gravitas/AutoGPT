import { describe, expect, test, vi } from "vitest";
import { render, screen, within } from "@/tests/integrations/test-utils";
import { InputExpanderModal } from "../TextInputExpanderModal";

vi.mock("@/lib/hooks/useBreakpoint", () => ({
  useBreakpoint: () => "lg",
  isLargeScreen: () => true,
}));

describe("InputExpanderModal", () => {
  test("renders dialog with provided title", () => {
    render(
      <InputExpanderModal
        isOpen={true}
        onClose={vi.fn()}
        onSave={vi.fn()}
        title="Custom Title"
        defaultValue="hello"
      />,
    );

    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByText("Custom Title")).toBeDefined();
  });

  test("renders dialog with fallback title when title is not provided", () => {
    render(
      <InputExpanderModal
        isOpen={true}
        onClose={vi.fn()}
        onSave={vi.fn()}
        defaultValue="hello"
      />,
    );

    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByText("Edit Text")).toBeDefined();
  });

  test("renders dialog with fallback title when title is empty string", () => {
    render(
      <InputExpanderModal
        isOpen={true}
        onClose={vi.fn()}
        onSave={vi.fn()}
        title=""
        defaultValue="hello"
      />,
    );

    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByText("Edit Text")).toBeDefined();
  });

  test("does not render dialog when closed", () => {
    render(
      <InputExpanderModal
        isOpen={false}
        onClose={vi.fn()}
        onSave={vi.fn()}
        defaultValue="hello"
      />,
    );

    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
