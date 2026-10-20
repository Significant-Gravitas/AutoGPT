import { describe, expect, test, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
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
    const heading = within(dialog).getByRole("heading", {
      name: "Drawer Title",
    });
    expect(heading).toBeDefined();
    expect(heading.classList.contains("sr-only")).toBe(false);
  });

  test("renders sr-only fallback title when no title prop is provided", () => {
    renderDrawerDialog({});

    const dialog = screen.getByRole("dialog");
    const fallback = within(dialog).getByRole("heading", { name: "Dialog" });
    expect(fallback).toBeDefined();
    expect(fallback.classList.contains("sr-only")).toBe(true);
  });

  test("renders sr-only fallback when title is empty string", () => {
    renderDrawerDialog({ title: "" });

    const dialog = screen.getByRole("dialog");
    const fallback = within(dialog).getByRole("heading", { name: "Dialog" });
    expect(fallback).toBeDefined();
    expect(fallback.classList.contains("sr-only")).toBe(true);
  });

  test("renders close button when not force open", () => {
    renderDrawerDialog({ title: "Test" });

    const dialog = screen.getByRole("dialog");
    const closeButton = within(dialog).getByLabelText("Close");
    expect(closeButton).toBeDefined();
  });

  test("leaves focus where it was on open by default", async () => {
    const opener = focusOpener();

    renderOpenDrawer({ autoFocus: false });

    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(document.activeElement).toBe(opener);
    opener.remove();
  });

  test("moves focus into the drawer on open with autoFocus", async () => {
    const opener = focusOpener();

    renderOpenDrawer({ autoFocus: true });

    const dialog = screen.getByRole("dialog");
    await waitFor(() =>
      expect(dialog.contains(document.activeElement)).toBe(true),
    );
    opener.remove();
  });
});

function focusOpener() {
  const opener = document.createElement("button");
  document.body.appendChild(opener);
  opener.focus();
  return opener;
}

function renderOpenDrawer({ autoFocus }: { autoFocus: boolean }) {
  return render(
    <Dialog
      title="Drawer"
      autoFocus={autoFocus}
      controlled={{ isOpen: true, set: vi.fn() }}
    >
      <Dialog.Content>
        <p>Drawer body</p>
      </Dialog.Content>
    </Dialog>,
  );
}
