import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { Copy01Icon, Tick02Icon } from "@hugeicons/core-free-icons";

import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { ShareRunButton } from "../ShareRunButton";

const SHARE_TOKEN = "550e8400-e29b-41d4-a716-446655440000";

const writeText = vi.fn(async (_text: string) => {});

let clipboardDescriptor: PropertyDescriptor | undefined;

beforeEach(() => {
  clipboardDescriptor = Object.getOwnPropertyDescriptor(navigator, "clipboard");
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { writeText },
  });
  writeText.mockClear();
});

afterEach(() => {
  vi.useRealTimers();
  if (clipboardDescriptor) {
    Object.defineProperty(navigator, "clipboard", clipboardDescriptor);
  } else {
    Reflect.deleteProperty(navigator, "clipboard");
  }
});

async function openShareDialog() {
  render(
    <ShareRunButton
      graphId="graph-1"
      executionId="run-1"
      isShared
      shareToken={SHARE_TOKEN}
    />,
  );

  fireEvent.click(screen.getByRole("button", { name: "Share results" }));
  return screen.findByRole("dialog", { name: "Share Workflow Run" });
}

// Which icon the button draws, by its geometry: both icons are unnamed SVGs.
function drawnPaths(element: HTMLElement) {
  return Array.from(element.querySelectorAll("svg path"), (path) =>
    path.getAttribute("d"),
  );
}

function pathsOf(icon: typeof Copy01Icon) {
  return icon.map(([, attributes]) => attributes.d);
}

describe("ShareRunButton", () => {
  test("names the dialog's copy button, which copies the share link", async () => {
    const dialog = await openShareDialog();
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Copy share link" }),
    );

    await waitFor(() =>
      expect(writeText).toHaveBeenCalledWith(
        `${window.location.origin}/share/${SHARE_TOKEN}`,
      ),
    );
  });

  test("shows a tick after a copy, and the copy icon again 2 s later", async () => {
    const dialog = await openShareDialog();
    const copyButton = within(dialog).getByRole("button", {
      name: "Copy share link",
    });
    expect(drawnPaths(copyButton)).toEqual(pathsOf(Copy01Icon));

    vi.useFakeTimers();
    await act(async () => {
      fireEvent.click(copyButton);
    });
    expect(writeText).toHaveBeenCalledOnce();
    expect(drawnPaths(copyButton)).toEqual(pathsOf(Tick02Icon));

    act(() => {
      vi.advanceTimersByTime(1999);
    });
    expect(drawnPaths(copyButton)).toEqual(pathsOf(Tick02Icon));

    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(drawnPaths(copyButton)).toEqual(pathsOf(Copy01Icon));
  });
});
