import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { showResult } from "./renderConnected";
import { tripSource } from "@/lib/openui/__tests__/capability-fixtures";

describe("one connected itinerary in chat", () => {
  beforeEach(() => sessionStorage.clear());
  it("links places, visit edits, completion, available time and submitted plan", async () => {
    const send = vi.fn();
    await showResult(send, false, tripSource);
    expect(await screen.findByText(/195 min remaining/)).toBeDefined();
    fireEvent.click(
      screen.getByRole("button", { name: "Show Museum visit on map" }),
    );
    expect(
      screen
        .getByRole("button", { name: "2. Museum" })
        .getAttribute("aria-pressed"),
    ).toBe("true");
    fireEvent.change(screen.getByLabelText("Museum visit minutes"), {
      target: { value: "abc" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    expect(send).not.toHaveBeenCalled();
    expect(
      (
        screen.getByRole("button", {
          name: "Download current calendar",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
    fireEvent.change(screen.getByLabelText("Museum visit minutes"), {
      target: { value: "120" },
    });
    fireEvent.click(screen.getByLabelText("Completed Golden Gate Bridge"));
    fireEvent.change(screen.getByLabelText("Saturday window 1 start"), {
      target: { value: "10:00" },
    });
    expect(
      await screen.findByText(/165 min remaining.*420 min available/),
    ).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(send).toHaveBeenCalledOnce());
    const data = JSON.parse(
      send.mock.calls[0][0].split("Submitted values:\n")[1],
    );
    expect(data.trip.tasks[0].done).toBe(true);
    expect(data.trip.tasks[1].minutes).toBe(120);
    expect(data.trip_required_minutes).toBe(165);
  });
  it("pauses after a date or route change invalidates supplied transfers", async () => {
    await showResult(vi.fn(), false, tripSource);
    fireEvent.click(
      await screen.findByRole("button", { name: "Move Museum visit earlier" }),
    );
    expect(screen.getByText(/needs a new travel estimate/)).toBeDefined();
    expect(
      (
        screen.getByRole("button", {
          name: "Download current calendar",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
    fireEvent.click(
      screen.getByRole("button", { name: "Move Museum visit later" }),
    );
    fireEvent.change(screen.getByLabelText("Saturday date"), {
      target: { value: "2026-10-18" },
    });
    expect(screen.getByText(/supplied for a different date/)).toBeDefined();
    expect(
      (
        screen.getByRole("button", {
          name: "Download current calendar",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
  });
});
