import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { showResult } from "./renderConnected";
import { formView } from "@/lib/openui/__tests__/capability-fixtures";

describe("structured choices and input boundaries", () => {
  beforeEach(() => sessionStorage.clear());
  it("retains selections across searches and blocks a required empty selection", async () => {
    const send = vi.fn();
    const options = Array.from({ length: 65 }, (_, index) => ({
      id: `guest${index + 1}`,
      label: `Guest ${index + 1}`,
      detail: "Supplied attendee",
    }));
    await showResult(
      send,
      false,
      formView(
        `SearchSelect("guests", "Guests", ${JSON.stringify(options)}, [], true, true)`,
      ),
    );
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    expect(send).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText("Search Guests"), {
      target: { value: "Guest 31" },
    });
    fireEvent.click(screen.getByRole("checkbox", { name: /Guest 31/ }));
    fireEvent.change(screen.getByLabelText("Search Guests"), {
      target: { value: "Guest 65" },
    });
    fireEvent.click(screen.getByRole("checkbox", { name: /Guest 65/ }));
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(send).toHaveBeenCalledOnce());
    expect(send.mock.calls[0][0]).toContain('"guest31"');
    expect(send.mock.calls[0][0]).toContain('"guest65"');
  });
  it("submits reordered identities and preserves them after reopening", async () => {
    const source = formView(
      'RankedList("priority", "Release order", [{id:"login",label:"Fix login",detail:""},{id:"export",label:"Add export",detail:""},{id:"search",label:"Improve search",detail:""}], ["login","export","search"])',
    );
    const send = vi.fn();
    const view = await showResult(send, false, source);
    fireEvent.click(
      screen.getByRole("button", { name: "Move Improve search up" }),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Move Improve search up" }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(send).toHaveBeenCalledOnce());
    expect(
      JSON.parse(send.mock.calls[0][0].split("Submitted values:\n")[1])
        .priority,
    ).toEqual(["search", "login", "export"]);
    view.unmount();
    await showResult(send, false, source);
    expect(screen.getByText("1. Improve search")).toBeDefined();
  });
  it("warns on inverted date/range edits and distinguishes unknown from no", async () => {
    const source = `root = Workspace("Callback", "", [form])
form = Form("callback", "Preferences", [date, budget, access], "Continue", "Use these preferences")
date = DateRangeField("stay", "Stay", {start:"2026-11-20",end:"2026-11-22"}, true)
budget = RangeField("budget", "Budget", {low:50,high:100}, 0, 1000, 1, "USD")
access = ValidatedField("access", {key:"access",label:"Step-free access",type:"boolean",required:true}, null)`;
    const send = vi.fn();
    await showResult(send, false, source);
    expect(
      (screen.getByLabelText("Step-free access") as HTMLSelectElement).value,
    ).toBe("");
    fireEvent.change(screen.getByLabelText("Step-free access"), {
      target: { value: "false" },
    });
    fireEvent.change(screen.getByLabelText("Stay end"), {
      target: { value: "2026-11-19" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    expect(send).not.toHaveBeenCalled();
    expect(screen.getByText(/End date must/)).toBeDefined();
    fireEvent.change(screen.getByLabelText("Stay end"), {
      target: { value: "2026-11-22" },
    });
    fireEvent.change(screen.getByLabelText("Budget low"), {
      target: { value: "letters" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    expect(send).not.toHaveBeenCalled();
    expect(screen.getByText(/Enter a number/)).toBeDefined();
    fireEvent.change(screen.getByLabelText("Budget low"), {
      target: { value: "101" },
    });
    expect(screen.getByText(/Lower bound must/)).toBeDefined();
    fireEvent.change(screen.getByLabelText("Budget low"), {
      target: { value: "75" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(send).toHaveBeenCalledOnce());
    const data = JSON.parse(
      send.mock.calls[0][0].split("Submitted values:\n")[1],
    );
    expect(data.access).toBe(false);
    expect(data.budget).toEqual({ low: 75, high: 100 });
  });
});
