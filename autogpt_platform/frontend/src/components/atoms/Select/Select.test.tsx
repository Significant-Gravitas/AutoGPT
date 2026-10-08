import { render, screen, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { Select } from "./Select";

function triggerText(name: string) {
  return within(screen.getByRole("combobox", { name })).getByText(/.+/, {
    selector: "[data-slot='select-value'], [data-slot='select-value'] *",
  }).textContent;
}

describe("Select", () => {
  test("runs an option's onSelect without changing the value", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    const onValueChange = vi.fn();
    render(
      <Select
        id="provider"
        label="Provider"
        onValueChange={onValueChange}
        options={[
          { value: "openai", label: "OpenAI" },
          { value: "add", label: "Add provider", onSelect },
        ]}
      />,
    );

    await user.click(screen.getByRole("combobox", { name: "Provider" }));
    await user.click(
      await screen.findByRole("option", { name: "Add provider" }),
    );

    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onValueChange).not.toHaveBeenCalled();
    expect(triggerText("Provider")).toBe("Provider");
  });

  test("reports a regular option through onValueChange", async () => {
    const user = userEvent.setup();
    const onValueChange = vi.fn();
    render(
      <Select
        id="provider"
        label="Provider"
        onValueChange={onValueChange}
        options={[{ value: "openai", label: "OpenAI" }]}
      />,
    );

    await user.click(screen.getByRole("combobox", { name: "Provider" }));
    await user.click(await screen.findByRole("option", { name: "OpenAI" }));

    expect(onValueChange).toHaveBeenCalledWith("openai");
    expect(triggerText("Provider")).toBe("OpenAI");
  });

  test("shows the selected option's label without opening the popup", () => {
    render(
      <Select
        id="provider"
        label="Provider"
        value="openai"
        options={[
          { value: "openai", label: "OpenAI" },
          { value: "anthropic", label: "Anthropic" },
        ]}
      />,
    );

    expect(triggerText("Provider")).toBe("OpenAI");
  });

  test("marks an error with aria-invalid", () => {
    render(
      <Select
        id="provider"
        label="Provider"
        error="Pick one"
        options={[{ value: "openai", label: "OpenAI" }]}
      />,
    );

    expect(
      screen
        .getByRole("combobox", { name: "Provider" })
        .getAttribute("aria-invalid"),
    ).toBe("true");
    expect(screen.getByText("Pick one")).toBeDefined();
  });

  test("keeps the label tooltip button outside the label element", () => {
    render(
      <Select
        id="provider"
        label="Provider"
        labelTooltip="Where the model runs"
        options={[{ value: "openai", label: "OpenAI" }]}
      />,
    );

    const tooltipButton = screen.getByRole("button", {
      name: "More information",
    });
    expect(tooltipButton.closest("label")).toBeNull();
    expect(screen.getByRole("combobox", { name: "Provider" })).toBeDefined();
  });
});
