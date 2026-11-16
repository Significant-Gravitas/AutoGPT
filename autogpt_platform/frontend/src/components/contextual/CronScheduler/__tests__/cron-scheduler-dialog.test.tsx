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
});
