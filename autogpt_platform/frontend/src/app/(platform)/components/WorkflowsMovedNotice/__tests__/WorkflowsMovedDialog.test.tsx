import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, test, vi } from "vitest";
import { WorkflowsMovedDialog } from "../WorkflowsMovedDialog";

function NoticeHarness({ onDismiss }: { onDismiss: () => void }) {
  const [isOpen, setIsOpen] = useState(false);

  function dismiss() {
    setIsOpen(false);
    onDismiss();
  }

  return (
    <>
      <button onClick={() => setIsOpen(true)}>Continue working</button>
      <WorkflowsMovedDialog isOpen={isOpen} onDismiss={dismiss} />
    </>
  );
}

describe("WorkflowsMovedDialog", () => {
  test("announces its title and explanation, with a readable destination", () => {
    render(<WorkflowsMovedDialog isOpen onDismiss={vi.fn()} />);

    const dialog = screen.getByRole("dialog", {
      name: "Your agents have moved",
    });
    const descriptionID = dialog.getAttribute("aria-describedby");
    expect(descriptionID).toBeTruthy();
    const description = document.getElementById(descriptionID!);
    expect(description?.textContent).toContain("Agents are now workflows");
    expect(description?.textContent).toContain("Otto");
    const destination = within(dialog).getByRole("list", {
      name: "Where to find your workflows",
    });
    expect(
      within(destination)
        .getAllByRole("listitem")
        .map((item) => item.textContent),
    ).toEqual(["Team", "Otto", "Workflows"]);
  });

  test("links directly to Otto's workflows and records acknowledgment on click", async () => {
    const user = userEvent.setup();
    const onDismiss = vi.fn();
    render(<WorkflowsMovedDialog isOpen onDismiss={onDismiss} />);

    const link = screen.getByRole("link", { name: "Go to my workflows" });
    expect(link.getAttribute("href")).toBe("/team/autopilot?tab=workflows");
    await user.click(link);

    expect(onDismiss).toHaveBeenCalledTimes(1);
  });

  test.each(["Got it", "Close notice"])(
    "allows dismissal with %s",
    async (name) => {
      const user = userEvent.setup();
      const onDismiss = vi.fn();
      render(<WorkflowsMovedDialog isOpen onDismiss={onDismiss} />);

      await user.click(screen.getByRole("button", { name }));

      expect(onDismiss).toHaveBeenCalledTimes(1);
    },
  );

  test("focuses the heading and contains keyboard focus", async () => {
    const user = userEvent.setup();
    render(<NoticeHarness onDismiss={vi.fn()} />);
    await user.click(screen.getByRole("button", { name: "Continue working" }));

    const dialog = screen.getByRole("dialog");
    expect(document.activeElement).toBe(
      screen.getByRole("heading", { name: "Your agents have moved" }),
    );
    await user.tab();
    expect(document.activeElement).toBe(
      screen.getByRole("button", { name: /play walkthrough/i }),
    );
    await user.tab();
    expect(document.activeElement).toBe(
      screen.getByRole("link", { name: "Go to my workflows" }),
    );
    await user.tab();
    expect(document.activeElement).toBe(
      screen.getByRole("button", { name: "Got it" }),
    );

    for (let index = 0; index < 6; index += 1) {
      await user.tab();
      expect(dialog.contains(document.activeElement)).toBe(true);
    }
    for (let index = 0; index < 4; index += 1) {
      await user.tab({ shift: true });
      expect(dialog.contains(document.activeElement)).toBe(true);
    }
  });

  test("Escape dismisses and returns focus to the previous task", async () => {
    const user = userEvent.setup();
    const onDismiss = vi.fn();
    render(<NoticeHarness onDismiss={onDismiss} />);
    const previousControl = screen.getByRole("button", {
      name: "Continue working",
    });
    await user.click(previousControl);

    await user.keyboard("{Escape}");

    expect(onDismiss).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("dialog")).toBeNull();
    await waitFor(() => expect(document.activeElement).toBe(previousControl));
  });

  test("does not dismiss when Escape belongs to an input method", () => {
    const onDismiss = vi.fn();
    render(<WorkflowsMovedDialog isOpen onDismiss={onDismiss} />);

    fireEvent.keyDown(screen.getByRole("dialog"), {
      key: "Escape",
      isComposing: true,
    });

    expect(onDismiss).not.toHaveBeenCalled();
  });

  test("renders no notice when closed", () => {
    render(<WorkflowsMovedDialog isOpen={false} onDismiss={vi.fn()} />);

    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.queryByText("Your agents have moved")).toBeNull();
  });
});
