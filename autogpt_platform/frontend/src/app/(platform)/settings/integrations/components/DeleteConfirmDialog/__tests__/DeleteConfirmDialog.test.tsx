import { describe, expect, test, vi } from "vitest";

import { fireEvent, render, screen } from "@/tests/integrations/test-utils";

import { DeleteConfirmDialog } from "../DeleteConfirmDialog";

describe("DeleteConfirmDialog", () => {
  test("does not render when closed", () => {
    render(
      <DeleteConfirmDialog
        open={false}
        onOpenChange={() => {}}
        itemNames={["GitHub key"]}
        onConfirm={() => {}}
      />,
    );
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  test("single delete: titles the dialog with the credential name", () => {
    render(
      <DeleteConfirmDialog
        open
        onOpenChange={() => {}}
        itemNames={["GitHub key"]}
        onConfirm={() => {}}
      />,
    );
    const dialog = screen.getByRole("dialog");
    expect(dialog.textContent).toContain("Remove GitHub key");
  });

  test("bulk delete: titles by count and lists names with overflow indicator", () => {
    render(
      <DeleteConfirmDialog
        open
        onOpenChange={() => {}}
        itemNames={["A", "B", "C", "D", "E"]}
        onConfirm={() => {}}
      />,
    );
    const dialog = screen.getByRole("dialog");
    expect(dialog.textContent).toContain("Remove 5 integrations?");
    expect(dialog.textContent).toContain("A, B, C");
    expect(dialog.textContent).toContain("and 2 more");
  });

  test("Remove button fires onConfirm; Cancel fires onOpenChange(false)", () => {
    const onConfirm = vi.fn();
    const onOpenChange = vi.fn();
    render(
      <DeleteConfirmDialog
        open
        onOpenChange={onOpenChange}
        itemNames={["GitHub"]}
        onConfirm={onConfirm}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /^remove$/i }));
    expect(onConfirm).toHaveBeenCalledOnce();

    fireEvent.click(screen.getByRole("button", { name: /cancel/i }));
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  test("force variant: retitles, swaps the warning copy, and labels the action 'Force remove'", () => {
    render(
      <DeleteConfirmDialog
        open
        variant="force"
        onOpenChange={() => {}}
        itemNames={["GitHub key"]}
        onConfirm={() => {}}
      />,
    );
    const dialog = screen.getByRole("dialog");
    expect(dialog.textContent).toContain("Force remove GitHub key");
    expect(dialog.textContent).toMatch(
      /referenced by an active webhook or workflow/i,
    );
    expect(
      screen.getByRole("button", { name: /^force remove$/i }),
    ).toBeDefined();
  });

  test("when isPending, both buttons are disabled", () => {
    const onConfirm = vi.fn();
    render(
      <DeleteConfirmDialog
        open
        isPending
        onOpenChange={() => {}}
        itemNames={["GitHub"]}
        onConfirm={onConfirm}
      />,
    );
    const removeBtn = screen.getByRole("button", { name: /^remove$/i });
    const cancelBtn = screen.getByRole("button", { name: /cancel/i });
    expect((removeBtn as HTMLButtonElement).disabled).toBe(true);
    expect((cancelBtn as HTMLButtonElement).disabled).toBe(true);
  });
});
