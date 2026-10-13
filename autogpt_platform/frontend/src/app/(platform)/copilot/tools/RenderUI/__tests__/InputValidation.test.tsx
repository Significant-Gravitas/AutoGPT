import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { render } from "@/tests/integrations/test-utils";
import { planning } from "@/lib/openui/__tests__/expanded-fixtures";
import { CopilotChatActionsProvider } from "../../../components/CopilotChatActionsProvider/CopilotChatActionsProvider";
import { RenderUI } from "../RenderUI";

function showForm(send = vi.fn(), source = planning) {
  return render(
    <CopilotChatActionsProvider onSend={send} chatSurface="copilot">
      <RenderUI
        part={{
          type: "tool-render_ui",
          toolCallId: "validation-ui",
          state: "output-available",
          input: {},
          output: {
            type: "ui_rendered",
            version: 1,
            session_id: "validation-session",
            source,
            message: "Visit planning summary",
          },
        }}
      />
    </CopilotChatActionsProvider>,
  );
}

describe("local generated form validation", () => {
  beforeEach(() => sessionStorage.clear());

  it("explains letters and pasted text without sending them to the model", async () => {
    const user = userEvent.setup();
    const send = vi.fn().mockResolvedValue(undefined);
    showForm(send);
    const budget = await screen.findByLabelText("Budget");
    await user.clear(budget);
    await user.paste("not a number");
    expect(screen.getByText(/Enter a number/)).toBeDefined();
    expect(document.activeElement).toBe(budget);
    expect(budget.getAttribute("aria-invalid")).toBe("true");
    expect(
      document.getElementById(budget.getAttribute("aria-describedby")!)
        ?.textContent,
    ).toContain("Enter a number");
    await user.click(screen.getByRole("button", { name: "Update plan" }));
    expect(send).not.toHaveBeenCalled();
    expect(document.activeElement).toBe(budget);
    await user.clear(budget);
    await user.type(budget, "240");
    await user.click(screen.getByRole("button", { name: "Update plan" }));
    await waitFor(() => expect(send).toHaveBeenCalledTimes(1));
    expect(send.mock.calls[0][0]).toContain('"budget": 240');
    expect(send.mock.calls[0][0]).not.toContain("not a number");
    expect(budget.getAttribute("aria-invalid")).toBe("false");
  });

  it.each([
    ["-10", /Enter 0 or more/],
    ["1100", /Enter 1000 or less/],
    ["105", /increments of 10/],
    ["1e999", /Enter a finite number/],
    ["", /Enter a number/],
  ])(
    "blocks an invalid budget %s even on direct form submission",
    async (value, message) => {
      const send = vi.fn();
      showForm(send);
      const budget = await screen.findByLabelText("Budget");
      fireEvent.change(budget, { target: { value } });
      fireEvent.submit(budget.closest("form")!);
      expect(screen.getByText(message)).toBeDefined();
      expect(send).not.toHaveBeenCalled();
    },
  );

  it("keeps an invalid draft on reload and clears its warning after correction", async () => {
    const first = showForm();
    fireEvent.change(await screen.findByLabelText("Budget"), {
      target: { value: "abc" },
    });
    first.unmount();
    showForm();
    const budget = await screen.findByLabelText("Budget");
    expect((budget as HTMLInputElement).value).toBe("abc");
    fireEvent.blur(budget);
    expect(screen.getByText(/Enter a number/)).toBeDefined();
    fireEvent.change(budget, { target: { value: "0" } });
    expect(screen.queryByText(/Enter a number/)).toBeNull();
  });

  it("shows a required date warning locally and accepts the corrected date", async () => {
    const send = vi.fn().mockResolvedValue(undefined);
    showForm(send);
    const date = await screen.findByLabelText("Visit date");
    fireEvent.change(date, { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: "Update plan" }));
    expect(screen.getByText(/Choose a valid date/)).toBeDefined();
    expect(send).not.toHaveBeenCalled();
    fireEvent.change(date, { target: { value: "2026-10-12" } });
    fireEvent.click(screen.getByRole("button", { name: "Update plan" }));
    await waitFor(() => expect(send).toHaveBeenCalledTimes(1));
  });

  it("renders in the message flow without presentation tabs or a framed scroller", async () => {
    showForm();
    const response = await screen.findByRole("region", {
      name: "Interactive response",
    });
    expect(screen.queryByRole("button", { name: "Explore" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Summary" })).toBeNull();
    expect(screen.queryByText("Interactive view")).toBeNull();
    expect(response.className).not.toMatch(/border|rounded|overflow/);
    expect(response.querySelector('[class*="max-h-"]')).toBeNull();
  });

  it("lets users type signed decimal values without losing the decimal point", async () => {
    const user = userEvent.setup();
    const send = vi.fn().mockResolvedValue(undefined);
    showForm(send, planning.replace("100, 0, 1000, 10)", "0, -10, 10, 0.5)"));
    const budget = await screen.findByLabelText("Budget");
    await user.clear(budget);
    await user.type(budget, "-1.5");
    expect((budget as HTMLInputElement).value).toBe("-1.5");
    await user.click(screen.getByRole("button", { name: "Update plan" }));
    await waitFor(() => expect(send).toHaveBeenCalledTimes(1));
    expect(send.mock.calls[0][0]).toContain('"budget": -1.5');
  });

  it("does not interrupt composing input with a warning", async () => {
    showForm();
    const budget = await screen.findByLabelText("Budget");
    fireEvent.blur(budget);
    fireEvent.compositionStart(budget);
    fireEvent.change(budget, { target: { value: "あ" } });
    expect(screen.queryByText(/Enter a number/)).toBeNull();
    fireEvent.compositionEnd(budget);
    expect(screen.getByText(/Enter a number/)).toBeDefined();
  });
});
