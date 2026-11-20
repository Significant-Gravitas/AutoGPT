import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { expect, test } from "vitest";
import { FullscreenDialog } from "../FullscreenDialog";

function Harness() {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <button onClick={() => setOpen(true)}>Edit Soul</button>
      <button>Background action</button>
      {open ? (
        <FullscreenDialog title="Maria's Soul" onClose={() => setOpen(false)}>
          <button onClick={() => setOpen(false)}>Close Soul</button>
          <input aria-label="Identity" />
          <button>Save Soul</button>
        </FullscreenDialog>
      ) : null}
    </div>
  );
}

test("focuses and traps the mobile panel, hides the background, and restores focus", async () => {
  render(<Harness />);
  const trigger = screen.getByRole("button", { name: "Edit Soul" });
  await userEvent.click(trigger);
  const dialog = await screen.findByRole("dialog", { name: "Maria's Soul" });
  await waitFor(() =>
    expect(dialog.contains(document.activeElement)).toBe(true),
  );
  expect(
    screen.queryByRole("button", { name: "Background action" }),
  ).toBeNull();
  const first = screen.getByRole("button", { name: "Close Soul" });
  const last = screen.getByRole("button", { name: "Save Soul" });
  last.focus();
  await userEvent.tab();
  expect(document.activeElement).toBe(first);
  await userEvent.tab({ shift: true });
  expect(document.activeElement).toBe(last);
  await userEvent.keyboard("{Escape}");
  await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  await waitFor(() => expect(document.activeElement).toBe(trigger));
});
