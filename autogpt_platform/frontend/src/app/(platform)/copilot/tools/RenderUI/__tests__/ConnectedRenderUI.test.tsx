import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { showResult } from "./renderConnected";
import { connected } from "@/lib/openui/__tests__/connected-fixtures";

describe("connected interactive plans in the conversation", () => {
  beforeEach(() => sessionStorage.clear());

  it("preserves cents after edits even when the model requests whole-dollar display", async () => {
    await showResult(
      vi.fn(),
      false,
      connected.replace('"USD", 2)', '"USD", 0)'),
    );
    fireEvent.click(
      await screen.findByRole("button", { name: "Choose Coastal cabin" }),
    );
    fireEvent.change(screen.getByLabelText("Museum tickets quantity"), {
      target: { value: "3" },
    });
    fireEvent.change(screen.getByLabelText("Museum tickets unit price"), {
      target: { value: "25.5" },
    });
    expect(await screen.findByText("$232.50")).toBeDefined();
  });

  it("connects a choice and editable costs to the total, then sends typed choices in the same chat", async () => {
    const send = vi.fn().mockResolvedValue(undefined);
    await showResult(send);
    fireEvent.click(
      await screen.findByRole("button", { name: "Choose Coastal cabin" }),
    );
    expect(await screen.findByText("$206.00")).toBeDefined();
    fireEvent.change(screen.getByLabelText("Museum tickets quantity"), {
      target: { value: "3" },
    });
    expect(await screen.findByText("$231.00")).toBeDefined();
    fireEvent.click(screen.getByLabelText("Include Cable car rides"));
    expect(await screen.findByText("$215.00")).toBeDefined();
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
    expect(message).toContain('"step_free": true');
    expect(message).toContain('"history"');
    expect(message).toContain('"quantity": 3');
    expect(message).toContain('"extras_total": 75');
    expect(message).toContain("No stairs.");
  });

  it("withholds a total for an invalid edit, blocks submission, and recovers without losing input", async () => {
    const send = vi.fn();
    await showResult(send);
    fireEvent.click(
      await screen.findByRole("button", { name: "Choose City hotel" }),
    );
    const price = screen.getByLabelText("Museum tickets unit price");
    fireEvent.focus(price);
    fireEvent.input(price, { target: { value: "abc" } });
    expect(await screen.findByText(/Enter a number/)).toBeDefined();
    expect(screen.queryByText("$246.00")).toBeNull();
    expect(
      screen.getByText("Check the highlighted inputs to calculate this total."),
    ).toBeDefined();
    fireEvent.click(
      screen.getByRole("button", { name: "Continue with this plan" }),
    );
    expect(send).not.toHaveBeenCalled();
    expect((price as HTMLInputElement).value).toBe("abc");
    fireEvent.change(price, { target: { value: "30" } });
    expect(await screen.findByText("$256.00")).toBeDefined();
  });

  it("recalculates a compound total when the number of nights changes and respects its constraints", async () => {
    await showResult();
    fireEvent.click(
      await screen.findByRole("button", { name: "Choose Coastal cabin" }),
    );
    const nights = screen.getByLabelText("Nights");
    fireEvent.change(nights, { target: { value: "3" } });
    expect(await screen.findByText("$486.00")).toBeDefined();
    fireEvent.change(nights, { target: { value: "0" } });
    fireEvent.blur(nights);
    expect(await screen.findByText("Enter 1 or more.")).toBeDefined();
    expect(screen.queryByText("$486.00")).toBeNull();
    expect(screen.queryByText("$66.00")).not.toBeNull();
  });

  it("restores selections, multi-selects and row edits after a remount and lets the user undo a choice", async () => {
    const first = await showResult();
    fireEvent.click(
      await screen.findByRole("button", { name: "Choose Coastal cabin" }),
    );
    fireEvent.click(screen.getByLabelText("History"));
    fireEvent.change(screen.getByLabelText("Museum tickets quantity"), {
      target: { value: "4" },
    });
    await screen.findByText("$256.00");
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
    expect(
      (screen.getByLabelText("Museum tickets quantity") as HTMLInputElement)
        .value,
    ).toBe("4");
    expect(await screen.findByText("$256.00")).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Choose City hotel" }));
    fireEvent.click(screen.getByRole("button", { name: "Undo choice" }));
    expect(
      screen
        .getByRole("button", { name: "Choose Coastal cabin" })
        .getAttribute("aria-pressed"),
    ).toBe("true");
  });

  it("keeps an excluded invalid draft local instead of sending a wrong numeric type", async () => {
    const send = vi.fn();
    await showResult(send);
    fireEvent.click(
      await screen.findByRole("button", { name: "Choose City hotel" }),
    );
    fireEvent.change(screen.getByLabelText("Museum tickets unit price"), {
      target: { value: "abc" },
    });
    fireEvent.click(screen.getByLabelText("Include Museum tickets"));
    fireEvent.click(
      screen.getByRole("button", { name: "Continue with this plan" }),
    );
    await waitFor(() => expect(send).toHaveBeenCalledTimes(1));
    expect(send.mock.calls[0][0]).not.toContain("abc");
    expect(send.mock.calls[0][0]).toContain('"included": false');
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
    expect(
      screen.getByLabelText("Museum tickets quantity").hasAttribute("disabled"),
    ).toBe(true);
    expect(screen.getByLabelText("History").closest("fieldset")?.disabled).toBe(
      true,
    );
  });
});
