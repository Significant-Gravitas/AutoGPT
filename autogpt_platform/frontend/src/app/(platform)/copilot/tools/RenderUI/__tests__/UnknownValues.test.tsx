import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { showResult } from "./renderConnected";

const unknown = `root = Workspace("Printer cost", "Enter the missing quote.", [form, total])
form = Form("quote", "Cedar quote", [NumberField("ink", "Monthly ink", null, 0, null, 0.01), CostTable("extras", "Extras", "USD", [{id:"delivery",label:"Delivery",quantity:1,unitPrice:null,included:true}])], "Compare costs", "Compare this quote")
total = CalculatedMetric("quote", "Annual cost", "90 + 12 * ink + extras_total", "currency", "USD", 2)`;

describe("unknown values in chat forms", () => {
  beforeEach(() => sessionStorage.clear());

  it("keeps missing prices blank, blocks submission, and calculates only after the user supplies them", async () => {
    const send = vi.fn();
    await showResult(send, false, unknown);
    const ink = await screen.findByLabelText("Monthly ink");
    const delivery = screen.getByLabelText("Delivery unit price");
    expect((ink as HTMLInputElement).value).toBe("");
    expect((delivery as HTMLInputElement).value).toBe("");
    expect(screen.queryByText("$90.00")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Compare costs" }));
    expect(send).not.toHaveBeenCalled();
    fireEvent.change(ink, { target: { value: "18.67" } });
    expect(screen.queryByText("$314.04")).toBeNull();
    fireEvent.change(delivery, { target: { value: "0" } });
    expect(await screen.findByText("$314.04")).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Compare costs" }));
    await waitFor(() => expect(send).toHaveBeenCalledTimes(1));
    expect(send.mock.calls[0][0]).toContain('"ink": 18.67');
    expect(send.mock.calls[0][0]).toContain('"unitPrice": 0');
  });

  it("preserves a cleared price after reload and supports deliberately excluding an unknown cost", async () => {
    const send = vi.fn();
    const first = await showResult(send, false, unknown);
    fireEvent.change(await screen.findByLabelText("Monthly ink"), {
      target: { value: "12" },
    });
    fireEvent.change(screen.getByLabelText("Delivery unit price"), {
      target: { value: "5" },
    });
    expect(await screen.findByText("$239.00")).toBeDefined();
    fireEvent.change(screen.getByLabelText("Delivery unit price"), {
      target: { value: "" },
    });
    await waitFor(() => expect(screen.queryByText("$239.00")).toBeNull());
    first.unmount();
    await showResult(send, false, unknown);
    expect(
      (
        (await screen.findByLabelText(
          "Delivery unit price",
        )) as HTMLInputElement
      ).value,
    ).toBe("");
    expect(screen.queryByText("$234.00")).toBeNull();
    fireEvent.click(screen.getByLabelText("Include Delivery"));
    expect(await screen.findByText("$234.00")).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Compare costs" }));
    await waitFor(() => expect(send).toHaveBeenCalledTimes(1));
    expect(send.mock.calls[0][0]).toContain('"included": false');
    expect(send.mock.calls[0][0]).not.toContain('"unitPrice": 0');
  });
});
