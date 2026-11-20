import { describe, expect, test, vi } from "vitest";
import {
  act,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";

import { SitrepItem } from "../SitrepItem";
import type { SitrepItemData } from "../../../types";

const ITEM: SitrepItemData = {
  id: "run-1",
  agentID: "agent-1",
  agentName: "Daily digest",
  agentImageUrl: "https://cdn.test/agent.png",
  priority: "success",
  message: "Finished a run",
  status: "healthy" as SitrepItemData["status"],
};

describe("SitrepItem", () => {
  test("renders the agent image while the URL loads", () => {
    render(<SitrepItem item={ITEM} />);
    expect(screen.getByAltText("Daily digest")).toBeDefined();
  });

  test("falls back to the priority icon when the image URL is dead", () => {
    render(<SitrepItem item={ITEM} />);

    fireEvent.error(screen.getByAltText("Daily digest"));

    expect(screen.queryByAltText("Daily digest")).toBeNull();
    expect(screen.getByText("Daily digest")).toBeDefined();
  });

  test("counts a running item's line up each second, and stops its clock when it goes", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-07-21T12:00:10Z"));
    try {
      const { unmount } = render(
        <SitrepItem
          item={{
            ...ITEM,
            priority: "running",
            message: "Running for 10s",
            runningSince: "2026-07-21T12:00:00Z",
            status: "running",
          }}
        />,
      );
      expect(screen.getByText("Running for 10s")).toBeDefined();

      act(() => {
        vi.advanceTimersByTime(6_000);
      });
      expect(screen.getByText("Running for 16s")).toBeDefined();

      unmount();
      expect(vi.getTimerCount()).toBe(0);
    } finally {
      vi.useRealTimers();
    }
  });

  test("starts no clock for an item that is not counting up", () => {
    vi.useFakeTimers();
    try {
      render(<SitrepItem item={ITEM} />);

      expect(screen.getByText("Finished a run")).toBeDefined();
      expect(vi.getTimerCount()).toBe(0);
    } finally {
      vi.useRealTimers();
    }
  });
});
