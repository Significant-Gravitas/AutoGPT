import { describe, expect, test, vi } from "vitest";
import { render, screen, within, fireEvent } from "@testing-library/react";
import { Dialog } from "../Dialog";

vi.mock("@/lib/hooks/useBreakpoint", () => ({
  useBreakpoint: () => "lg",
  isLargeScreen: () => true,
}));

function renderDialog({
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
        <p>Dialog body</p>
      </Dialog.Content>
    </Dialog>,
  );
}

describe("Dialog", () => {
  test("renders visible title when title prop is provided", () => {
    renderDialog({ title: "My Title" });

    const dialog = screen.getByRole("dialog");
    const heading = within(dialog).getByText("My Title");
    expect(heading).toBeDefined();
    expect(heading.classList.contains("sr-only")).toBe(false);
  });

  test("renders sr-only fallback title when no title prop is provided", () => {
    renderDialog({});

    const dialog = screen.getByRole("dialog");
    const fallback = within(dialog).getByText("Dialog");
    expect(fallback).toBeDefined();
    expect(fallback.classList.contains("sr-only")).toBe(true);
  });

  test("renders sr-only fallback title when title is empty string", () => {
    renderDialog({ title: "" });

    const dialog = screen.getByRole("dialog");
    const fallback = within(dialog).getByText("Dialog");
    expect(fallback).toBeDefined();
    expect(fallback.classList.contains("sr-only")).toBe(true);
  });

  test("renders close button when not force open", () => {
    renderDialog({ title: "Test" });

    const dialog = screen.getByRole("dialog");
    const closeButton = within(dialog).getByLabelText("Close");
    expect(closeButton).toBeDefined();
  });

  test("calls controlled.set(false) when close button is clicked", () => {
    const set = vi.fn();
    renderDialog({ title: "Test", controlled: { isOpen: true, set } });

    const dialog = screen.getByRole("dialog");
    const closeButton = within(dialog).getByLabelText("Close");
    fireEvent.click(closeButton);

    expect(set).toHaveBeenCalledWith(false);
  });

  test("does not render dialog when controlled isOpen is false", () => {
    renderDialog({
      title: "Test",
      controlled: { isOpen: false, set: vi.fn() },
    });

    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
