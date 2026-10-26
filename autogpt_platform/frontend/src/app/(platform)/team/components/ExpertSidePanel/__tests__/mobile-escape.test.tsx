import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, describe, expect, test, vi } from "vitest";
import { ExpertSidePanel } from "../ExpertSidePanel";

const viewport = { isMobile: true };

vi.mock("@/app/(platform)/copilot/useIsMobile", () => ({
  useIsMobile: () => viewport.isMobile,
}));
vi.mock("../usePanelSidebarCollapse", () => ({
  usePanelSidebarCollapse: () => undefined,
}));

function Fixture({ onClose }: { onClose: () => void }) {
  const [open, setOpen] = useState(true);
  return (
    <>
      <button type="button" onClick={() => setOpen(true)}>
        Open panel
      </button>
      <ExpertSidePanel
        identity={
          open ? { name: "Sample Expert", avatarUrl: null, color: null } : null
        }
        title="sample-skill"
        panelId="sample"
        closeLabel="Close sample panel"
        onClose={() => {
          onClose();
          setOpen(false);
        }}
      >
        <p>Panel body</p>
      </ExpertSidePanel>
    </>
  );
}

afterEach(() => {
  viewport.isMobile = true;
});

describe("ExpertSidePanel on a phone", () => {
  test("Escape closes the dialog even when the close button holds focus, and focus returns", async () => {
    const onClose = vi.fn();
    render(<Fixture onClose={onClose} />);
    // While the dialog is open the rest of the page is aria-hidden.
    const opener = screen.getByRole("button", {
      name: "Open panel",
      hidden: true,
    });
    const close = await screen.findByRole("button", {
      name: "Close sample panel",
    });
    // The dialog moves focus to the close button on open; the real Button
    // would open a hover tooltip on focus, which then swallows Escape.
    close.focus();
    expect(document.activeElement).toBe(close);
    expect(screen.queryByRole("tooltip")).toBeNull();
    await userEvent.keyboard("{Escape}");
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: "Close sample panel" }),
      ).toBeNull(),
    );
    expect(opener.isConnected).toBe(true);
  });

  test("the desktop close button keeps its tooltip and label", async () => {
    viewport.isMobile = false;
    render(<Fixture onClose={vi.fn()} />);
    const close = await screen.findByRole("button", {
      name: "Close sample panel",
    });
    close.focus();
    expect(
      (await screen.findAllByText("Close sample panel")).length,
    ).toBeGreaterThan(0);
  });
});
