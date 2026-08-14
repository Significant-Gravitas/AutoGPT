import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import {
  cleanup,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import ExpertPreviewPage from "../page";
import { chunkIntoRows, getProfessionImageSrc, PROFESSIONS } from "../helpers";

const motionState = vi.hoisted(() => ({ prefersReducedMotion: false }));

vi.mock("framer-motion", async () => {
  const React = await import("react");

  const MotionDiv = React.forwardRef<
    HTMLDivElement,
    React.HTMLAttributes<HTMLDivElement> & {
      animate?: unknown;
      exit?: unknown;
      initial?: unknown;
      layoutId?: string;
      transition?: unknown;
    }
  >(function MotionDiv(
    { animate, exit, initial, layoutId, transition, ...props },
    ref,
  ) {
    return React.createElement("div", {
      ...props,
      ref,
      "data-layout-id": layoutId,
      "data-motion-animate": JSON.stringify(animate),
      "data-motion-exit": JSON.stringify(exit),
      "data-motion-initial": JSON.stringify(initial),
      "data-motion-transition": JSON.stringify(transition),
    });
  });

  return {
    AnimatePresence: ({ children }: { children: React.ReactNode }) => children,
    motion: { div: MotionDiv },
    useReducedMotion: () => motionState.prefersReducedMotion,
  };
});

beforeEach(() => {
  motionState.prefersReducedMotion = false;
});

afterEach(() => {
  cleanup();
});

describe("expert preview helpers", () => {
  test("keeps a unique 50-profession catalog and splits every entry once", () => {
    expect(PROFESSIONS).toHaveLength(50);
    expect(new Set(PROFESSIONS.map(({ slug }) => slug))).toHaveLength(50);

    const rows = chunkIntoRows(PROFESSIONS, 4);

    expect(rows.map((row) => row.length)).toEqual([13, 13, 13, 11]);
    expect(rows.flat()).toEqual(PROFESSIONS);
    expect(getProfessionImageSrc("product_designer")).toBe(
      "/experts/professions/product_designer.webp",
    );
  });
});

describe("ExpertPreviewPage", () => {
  test("renders two seamless copies of every profession across four rows", () => {
    render(<ExpertPreviewPage />);

    expect(
      screen.getByRole("heading", { name: "Expert Preview" }),
    ).toBeDefined();
    expect(screen.getAllByRole("button")).toHaveLength(100);
    expect(
      screen.getAllByRole("button", { name: "View Marketing Strategist" }),
    ).toHaveLength(2);
  });

  test("traps focus in the dialog and restores it after Escape", async () => {
    const user = userEvent.setup();
    render(<ExpertPreviewPage />);
    const trigger = screen.getAllByRole("button", {
      name: "View Marketing Strategist",
    })[0];

    await user.click(trigger);
    const dialog = await screen.findByRole("dialog", {
      name: "Marketing Strategist",
    });

    await waitFor(() =>
      expect(dialog.contains(document.activeElement)).toBe(true),
    );
    await user.tab();
    expect(dialog.contains(document.activeElement)).toBe(true);
    await user.tab();
    expect(dialog.contains(document.activeElement)).toBe(true);
    await user.tab({ shift: true });
    expect(dialog.contains(document.activeElement)).toBe(true);
    await user.tab({ shift: true });
    expect(dialog.contains(document.activeElement)).toBe(true);

    await user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(document.activeElement).toBe(trigger);
  });

  test("closes on backdrop click and restores the duplicated trigger", async () => {
    const user = userEvent.setup();
    render(<ExpertPreviewPage />);
    const triggers = screen.getAllByRole("button", {
      name: "View Creative Director",
    });
    const trigger = triggers[1];

    await user.click(trigger);
    await screen.findByRole("dialog", {
      name: "Creative Director",
    });
    await user.click(screen.getByTestId("expert-preview-backdrop"));

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(document.activeElement).toBe(trigger);
  });

  test("removes layout, backdrop, image, and label motion when requested", async () => {
    motionState.prefersReducedMotion = true;
    const user = userEvent.setup();
    render(<ExpertPreviewPage />);

    await user.click(
      screen.getAllByRole("button", { name: "View Product Designer" })[0],
    );
    const dialog = await screen.findByRole("dialog", {
      name: "Product Designer",
    });
    const imageMotion = screen
      .getByRole("img", { name: "Product Designer" })
      .closest("[data-motion-transition]");
    const labelMotion = screen
      .getByRole("heading", { name: "Product Designer" })
      .closest("[data-motion-transition]");

    expect(dialog.getAttribute("data-motion-initial")).toBe("false");
    expect(dialog.getAttribute("data-motion-exit")).toBe('{"opacity":1}');
    expect(dialog.getAttribute("data-motion-transition")).toBe(
      '{"duration":0}',
    );
    expect(imageMotion?.getAttribute("data-layout-id")).toBeNull();
    expect(imageMotion?.getAttribute("data-motion-transition")).toBe(
      '{"duration":0}',
    );
    expect(labelMotion?.getAttribute("data-motion-initial")).toBe("false");
    expect(labelMotion?.getAttribute("data-motion-transition")).toBe(
      '{"duration":0}',
    );
  });
});
