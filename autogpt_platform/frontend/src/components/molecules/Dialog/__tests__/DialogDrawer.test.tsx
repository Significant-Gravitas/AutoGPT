import { describe, expect, test, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { Dialog } from "../Dialog";

vi.mock("@/lib/hooks/useBreakpoint", () => ({
  useBreakpoint: () => "sm",
  isLargeScreen: () => false,
}));

function renderDrawerDialog({
  title,
  controlled,
}: {
  title?: string;
  controlled?: { isOpen: boolean; set: (open: boolean) => void };
}) {
  return render(
    <Dialog
      title={title}
      controlled={controlled ?? { isOpen: true, set: vi.fn() }}
    >
      <Dialog.Content>
        <p>Drawer body</p>
      </Dialog.Content>
    </Dialog>,
  );
}

describe("Dialog as Drawer (small screen)", () => {
  test("renders visible title when title prop is provided", () => {
    renderDrawerDialog({ title: "Drawer Title" });

    const dialog = screen.getByRole("dialog");
    const heading = within(dialog).getByText("Drawer Title");
    expect(heading).toBeDefined();
    expect(heading.classList.contains("sr-only")).toBe(false);
  });

  test("renders sr-only fallback title when no title prop is provided", () => {
    renderDrawerDialog({});

    const dialog = screen.getByRole("dialog");
    const fallback = within(dialog).getByText("Dialog");
    expect(fallback).toBeDefined();
    expect(fallback.classList.contains("sr-only")).toBe(true);
  });

  test("renders sr-only fallback when title is empty string", () => {
    renderDrawerDialog({ title: "" });

    const dialog = screen.getByRole("dialog");
    const fallback = within(dialog).getByText("Dialog");
    expect(fallback).toBeDefined();
    expect(fallback.classList.contains("sr-only")).toBe(true);
  });

  test("renders close button when not force open", () => {
    renderDrawerDialog({ title: "Test" });

    const dialog = screen.getByRole("dialog");
    const closeButton = within(dialog).getByLabelText("Close");
    expect(closeButton).toBeDefined();
  });
});
