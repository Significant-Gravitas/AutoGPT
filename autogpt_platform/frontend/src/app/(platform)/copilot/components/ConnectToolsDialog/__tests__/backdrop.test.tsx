import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it, vi } from "vitest";
import { ConnectToolsDialog } from "../ConnectToolsDialog";

function renderDialog() {
  const onClose = vi.fn();
  render(<ConnectToolsDialog isOpen onClose={onClose} />);
  const dialog = screen.getByRole("dialog", { name: "Connect your tools" });
  const backdrop = dialog.parentElement as HTMLElement;
  return { onClose, dialog, backdrop };
}

describe("Connect your tools — the backdrop", () => {
  it("closes on a press that starts and ends on the backdrop", () => {
    const { onClose, backdrop } = renderDialog();

    fireEvent.pointerDown(backdrop);
    fireEvent.click(backdrop);

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("stays open when a text selection drags out of the dialog onto it", () => {
    // Selecting an API key and letting go past the dialog's edge fires the
    // click on the backdrop, the common ancestor; the half-typed key stays.
    const { onClose, dialog, backdrop } = renderDialog();

    fireEvent.pointerDown(dialog);
    fireEvent.click(backdrop);

    expect(onClose).not.toHaveBeenCalled();
  });
});
