import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { showResult } from "./renderConnected";

const unknown = `root = Workspace("Printer cost", "Enter the missing quote.", [form])
form = Form("quote", "Cedar quote", [NumberField("ink", "Monthly ink", null, 0, null, 0.01)], "Compare costs", "Compare this quote")`;

describe("unknown values in chat forms", () => {
  beforeEach(() => sessionStorage.clear());

  it("keeps a missing number blank and blocks submission until the user supplies it", async () => {
    const send = vi.fn();
    await showResult(send, false, unknown);
    const ink = await screen.findByLabelText("Monthly ink");
    expect((ink as HTMLInputElement).value).toBe("");
    fireEvent.click(screen.getByRole("button", { name: "Compare costs" }));
    expect(send).not.toHaveBeenCalled();
    fireEvent.change(ink, { target: { value: "18.67" } });
    fireEvent.click(screen.getByRole("button", { name: "Compare costs" }));
    await waitFor(() => expect(send).toHaveBeenCalledTimes(1));
    expect(send.mock.calls[0][0]).toContain('"ink": 18.67');
  });
});
