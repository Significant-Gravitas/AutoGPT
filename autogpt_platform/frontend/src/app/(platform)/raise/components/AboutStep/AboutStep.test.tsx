import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { AboutStep } from "./AboutStep";

describe("AboutStep", () => {
  test("shows the submitted about text in the answer bubble", () => {
    render(
      <AboutStep
        submittedAbout="Focus on practical SEO wins."
        name="Nova"
        color="rose-300"
        onSubmit={vi.fn()}
        onSkip={vi.fn()}
      />,
    );

    expect(screen.getByText("Focus on practical SEO wins.")).toBeDefined();
  });

  test("shows Skipped when the about step was skipped", () => {
    render(
      <AboutStep
        submittedAbout=""
        name="Nova"
        color="rose-300"
        onSubmit={vi.fn()}
        onSkip={vi.fn()}
      />,
    );

    expect(screen.getByText("Skipped")).toBeDefined();
    expect(screen.queryByRole("textbox")).toBeNull();
  });

  test("submits trimmed about text", async () => {
    const onSubmit = vi.fn();
    render(
      <AboutStep
        submittedAbout={null}
        name="Nova"
        color={null}
        onSubmit={onSubmit}
        onSkip={vi.fn()}
      />,
    );

    await userEvent.type(
      screen.getByRole("textbox", { name: "Anything about your expert" }),
      "  Ship weekly updates.  ",
    );
    await userEvent.click(screen.getByRole("button", { name: "That's it" }));

    expect(onSubmit).toHaveBeenCalledWith("Ship weekly updates.");
  });

  test("shows a generic placeholder when the expert has no name", () => {
    render(
      <AboutStep
        submittedAbout={null}
        name={null}
        color={null}
        onSubmit={vi.fn()}
        onSkip={vi.fn()}
      />,
    );

    expect(
      screen.getByPlaceholderText(
        "How they should work, what you care about, anything that helps them sound like yours…",
      ),
    ).toBeDefined();
  });

  test("personalizes the placeholder with the expert name", () => {
    render(
      <AboutStep
        submittedAbout={null}
        name="Nova"
        color={null}
        onSubmit={vi.fn()}
        onSkip={vi.fn()}
      />,
    );

    expect(
      screen.getByPlaceholderText(
        "How Nova should work, what you care about, anything that helps them sound like yours…",
      ),
    ).toBeDefined();
  });
});
