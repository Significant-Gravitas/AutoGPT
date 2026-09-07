import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, test, vi } from "vitest";
import { Dialog } from "../Dialog";

// Below the lg breakpoint the same <Dialog> renders as a drawer. Escape has to
// behave the same in both, which it did not before: only DialogWrap carried the
// composition guard.
vi.mock("@/lib/hooks/useBreakpoint", () => ({
  useBreakpoint: () => "sm",
  isLargeScreen: () => false,
}));

function renderDrawer({
  forceOpen,
  set,
}: {
  forceOpen?: boolean;
  set: (open: boolean) => void;
}) {
  return render(
    <Dialog
      title="Test"
      forceOpen={forceOpen}
      controlled={{ isOpen: true, set }}
    >
      <Dialog.Content>
        <p>Drawer body</p>
      </Dialog.Content>
    </Dialog>,
  );
}

describe("Dialog rendered as a drawer", () => {
  test("renders the drawer variant", () => {
    renderDrawer({ set: vi.fn() });

    expect(screen.getByText("Drawer body")).toBeDefined();
  });

  test("does not close when Escape belongs to an IME composition", () => {
    const set = vi.fn();
    renderDrawer({ set });

    fireEvent.keyDown(document, { key: "Escape", isComposing: true });

    expect(set).not.toHaveBeenCalled();
  });

  test("keeps a force-open drawer open on Escape", () => {
    const set = vi.fn();
    renderDrawer({ forceOpen: true, set });

    fireEvent.keyDown(document, { key: "Escape" });

    expect(set).not.toHaveBeenCalled();
  });

  test("closes on a plain Escape keydown", () => {
    const set = vi.fn();
    renderDrawer({ set });

    fireEvent.keyDown(document, { key: "Escape" });

    expect(set).toHaveBeenCalledWith(false);
  });
});
