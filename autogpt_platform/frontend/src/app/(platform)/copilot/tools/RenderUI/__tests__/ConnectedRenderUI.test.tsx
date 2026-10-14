import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { showResult } from "./renderConnected";

describe("connected interactive plans in the conversation", () => {
  beforeEach(() => sessionStorage.clear());

  it("sends typed choices and the chosen option's amount in the same chat", async () => {
    const send = vi.fn().mockResolvedValue(undefined);
    await showResult(send);
    fireEvent.click(
      await screen.findByRole("button", { name: "Choose Coastal cabin" }),
    );
    fireEvent.click(screen.getByLabelText("Step-free access needed"));
    fireEvent.click(screen.getByLabelText("History"));
    fireEvent.change(screen.getByLabelText("Anything else?"), {
      target: { value: "Keep the bridge in the plan.\nNo stairs." },
    });
    expect(send).not.toHaveBeenCalled();
    fireEvent.click(
      screen.getByRole("button", { name: "Continue with this plan" }),
    );
    await waitFor(() => expect(send).toHaveBeenCalledTimes(1));
    const message = send.mock.calls[0][0];
    expect(message).toContain('"stay": "coast"');
    expect(message).toContain('"stay_amount": 140');
    expect(message).toContain('"step_free": true');
    expect(message).toContain('"history"');
    expect(message).toContain("No stairs.");
  });

  it("enforces number constraints before submission", async () => {
    const send = vi.fn();
    await showResult(send);
    fireEvent.click(
      await screen.findByRole("button", { name: "Choose Coastal cabin" }),
    );
    const nights = screen.getByLabelText("Nights");
    fireEvent.change(nights, { target: { value: "0" } });
    fireEvent.blur(nights);
    expect(await screen.findByText("Enter 1 or more.")).toBeDefined();
    fireEvent.click(
      screen.getByRole("button", { name: "Continue with this plan" }),
    );
    expect(send).not.toHaveBeenCalled();
  });

  it("restores selections and multi-selects after a remount and lets the user undo a choice", async () => {
    const first = await showResult();
    fireEvent.click(
      await screen.findByRole("button", { name: "Choose Coastal cabin" }),
    );
    fireEvent.click(screen.getByLabelText("History"));
    first.unmount();
    await showResult();
    expect(
      (
        await screen.findByRole("button", { name: "Choose Coastal cabin" })
      ).getAttribute("aria-pressed"),
    ).toBe("true");
    expect((screen.getByLabelText("History") as HTMLInputElement).checked).toBe(
      true,
    );
    fireEvent.click(screen.getByRole("button", { name: "Choose City hotel" }));
    fireEvent.click(screen.getByRole("button", { name: "Undo choice" }));
    expect(
      screen
        .getByRole("button", { name: "Choose Coastal cabin" })
        .getAttribute("aria-pressed"),
    ).toBe("true");
  });

  it("requires a deliberate choice and does not allow shared views to edit or submit", async () => {
    const send = vi.fn();
    const first = await showResult(send);
    fireEvent.click(
      await screen.findByRole("button", { name: "Continue with this plan" }),
    );
    expect(send).not.toHaveBeenCalled();
    expect(
      await screen.findByText("Choose one option to continue."),
    ).toBeDefined();
    first.unmount();
    await showResult(send, true);
    expect(
      (
        await screen.findByRole("button", { name: "Choose City hotel" })
      ).hasAttribute("disabled"),
    ).toBe(true);
    expect(screen.getByLabelText("Nights").hasAttribute("disabled")).toBe(true);
    expect(screen.getByLabelText("History").closest("fieldset")?.disabled).toBe(
      true,
    );
  });
});
