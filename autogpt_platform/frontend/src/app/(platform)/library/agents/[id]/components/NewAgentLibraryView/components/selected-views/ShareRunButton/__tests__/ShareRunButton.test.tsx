import { beforeEach, describe, expect, test, vi } from "vitest";

import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { ShareRunButton } from "../ShareRunButton";

const SHARE_TOKEN = "550e8400-e29b-41d4-a716-446655440000";

const writeText = vi.fn(async (_text: string) => {});

beforeEach(() => {
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { writeText },
  });
  writeText.mockClear();
});

describe("ShareRunButton", () => {
  test("names the dialog's copy button, which copies the share link", async () => {
    render(
      <ShareRunButton
        graphId="graph-1"
        executionId="run-1"
        isShared
        shareToken={SHARE_TOKEN}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Share results" }));
    const dialog = await screen.findByRole("dialog", {
      name: "Share Workflow Run",
    });
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Copy share link" }),
    );

    await waitFor(() =>
      expect(writeText).toHaveBeenCalledWith(
        `${window.location.origin}/share/${SHARE_TOKEN}`,
      ),
    );
  });
});
