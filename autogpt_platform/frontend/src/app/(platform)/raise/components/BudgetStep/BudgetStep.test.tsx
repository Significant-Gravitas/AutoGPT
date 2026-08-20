import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { BudgetStep } from "./BudgetStep";

describe("BudgetStep", () => {
  test("shows Skipped when the budget step was skipped", () => {
    render(
      <BudgetStep
        color="rose-300"
        submittedBudget={{ credits: null }}
        onSubmit={vi.fn()}
        onSkip={vi.fn()}
      />,
    );

    expect(screen.getByText("Skipped")).toBeDefined();
    expect(screen.queryByRole("group", { name: "Weekly budget" })).toBeNull();
  });

  test("submits a preset immediately", async () => {
    const onSubmit = vi.fn();
    render(
      <BudgetStep
        color="rose-300"
        submittedBudget={null}
        onSubmit={onSubmit}
        onSkip={vi.fn()}
      />,
    );

    await userEvent.click(screen.getByRole("button", { name: /500 credits/ }));
    expect(onSubmit).toHaveBeenCalledWith(500);
  });

  test("submits a custom credit amount", async () => {
    const onSubmit = vi.fn();
    render(
      <BudgetStep
        color="rose-300"
        submittedBudget={null}
        onSubmit={onSubmit}
        onSkip={vi.fn()}
      />,
    );

    await userEvent.type(
      screen.getByRole("textbox", { name: "Custom weekly budget in credits" }),
      "750",
    );
    await userEvent.click(screen.getByRole("button", { name: "That's it" }));
    expect(onSubmit).toHaveBeenCalledWith(750);
  });

  test("skip keeps the platform default", async () => {
    const onSkip = vi.fn();
    render(
      <BudgetStep
        color="rose-300"
        submittedBudget={null}
        onSubmit={vi.fn()}
        onSkip={onSkip}
      />,
    );

    await userEvent.click(screen.getByRole("button", { name: "Skip" }));
    expect(onSkip).toHaveBeenCalled();
  });
});
