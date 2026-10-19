import { act, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ScheduledSendChip } from "../ScheduledSendChip";

const NOW = new Date(2026, 9, 12, 1, 0);

function chip(runAt: Date) {
  return (
    <ScheduledSendChip
      job={{ kind: "send", run_at: runAt, message: "Draft the launch post" }}
      onEdit={() => {}}
      onCancel={() => {}}
    />
  );
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(NOW);
});

afterEach(() => {
  vi.useRealTimers();
});

describe("ScheduledSendChip", () => {
  it("offers Edit and Cancel until the send is due", () => {
    render(chip(new Date(2026, 9, 12, 1, 7)));
    expect(screen.getByText("Sends at 1:07 AM")).toBeDefined();
    expect(screen.getByRole("button", { name: "Edit" })).toBeDefined();
  });

  it("shows it going, with no Edit, once it is due", () => {
    render(chip(new Date(2026, 9, 12, 1, 7)));
    act(() => {
      vi.advanceTimersByTime(7 * 60_000);
    });
    expect(screen.getByText("Sending…")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Edit" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Cancel" })).toBeNull();
  });

  it("is due straight away when its time has already passed", () => {
    render(chip(new Date(2026, 9, 12, 0, 59)));
    expect(screen.getByText("Sending…")).toBeDefined();
  });

  it("goes at its new time when the send moves while the chip is up", () => {
    const view = render(chip(new Date(2026, 9, 12, 1, 7)));
    act(() => {
      vi.advanceTimersByTime(5 * 60_000);
    });
    view.rerender(chip(new Date(2026, 9, 12, 1, 10)));
    expect(screen.getByText("Sends at 1:10 AM")).toBeDefined();
    act(() => {
      vi.advanceTimersByTime(5 * 60_000);
    });
    expect(screen.getByText("Sending…")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Edit" })).toBeNull();
  });
});
