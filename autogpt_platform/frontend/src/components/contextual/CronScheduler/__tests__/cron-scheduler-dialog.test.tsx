import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";

import { fireEvent, render, screen } from "@/tests/integrations/test-utils";

import { CronExpressionDialog } from "../cron-scheduler-dialog";

describe("CronExpressionDialog", () => {
  test("Cancel closes the dialog without submitting", async () => {
    const setOpen = vi.fn();
    const onSubmit = vi.fn();

    render(<CronExpressionDialog open setOpen={setOpen} onSubmit={onSubmit} />);
    fireEvent.click(await screen.findByRole("button", { name: "Cancel" }));

    expect(setOpen).toHaveBeenCalledWith(false);
    expect(onSubmit).not.toHaveBeenCalled();
  });

  test("a custom interval keeps what is typed, is flagged when out of range, and submits only once it is in range (#15275)", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<CronExpressionDialog open setOpen={vi.fn()} onSubmit={onSubmit} />);

    const [repeat] = await screen.findAllByRole("combobox");
    await user.click(repeat);
    await user.click(await screen.findByRole("option", { name: "Custom" }));
    const field = screen.getByRole<HTMLInputElement>("spinbutton");

    await user.clear(field);
    expect(field.value).toBe("");
    expect(screen.queryByText("Enter 1 to 60 minutes")).toBeNull();
    await user.click(screen.getByRole("button", { name: "Done" }));
    expect(field.getAttribute("aria-invalid")).toBe("true");
    expect(field.getAttribute("aria-describedby")).toBe(
      screen.getByText("Enter 1 to 60 minutes").id,
    );
    expect(onSubmit).not.toHaveBeenCalled();

    await user.type(field, "90");
    expect(field.value).toBe("90");
    expect(screen.getByText("Enter 1 to 60 minutes")).toBeDefined();
    await user.click(screen.getByRole("button", { name: "Done" }));
    expect(onSubmit).not.toHaveBeenCalled();

    await user.clear(field);
    await user.type(field, "60");
    await user.click(screen.getByRole("button", { name: "Done" }));
    expect(onSubmit).toHaveBeenCalledExactlyOnceWith("0 * * * *");
  });

  test("flags a prefilled out-of-range interval when it opens", async () => {
    render(
      <CronExpressionDialog
        open
        setOpen={vi.fn()}
        onSubmit={vi.fn()}
        defaultCronExpression="*/90 * * * *"
      />,
    );

    expect(await screen.findByText("Enter 1 to 60 minutes")).toBeDefined();
    expect(screen.getByRole<HTMLInputElement>("spinbutton").value).toBe("90");
  });
});
