import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, test, vi } from "vitest";
import { Select } from "./Select";

function openSelect(name: string) {
  fireEvent.click(screen.getByRole("combobox", { name }));
}

describe("Select", () => {
  test("runs an option's onSelect from the keyboard without changing the value", async () => {
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

    openSelect("Provider");
    const action = await screen.findByRole("option", { name: "Add provider" });
    fireEvent.keyDown(action, { key: "Enter" });

    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onValueChange).not.toHaveBeenCalled();
    expect(screen.getByRole("combobox", { name: "Provider" }).textContent).toBe(
      "Provider",
    );
  });

  test("reports a regular option through onValueChange", async () => {
    const onValueChange = vi.fn();
    render(
      <Select
        id="provider"
        label="Provider"
        onValueChange={onValueChange}
        options={[{ value: "openai", label: "OpenAI" }]}
      />,
    );

    openSelect("Provider");
    fireEvent.keyDown(await screen.findByRole("option", { name: "OpenAI" }), {
      key: "Enter",
    });

    expect(onValueChange).toHaveBeenCalledWith("openai");
    expect(screen.getByRole("combobox", { name: "Provider" }).textContent).toBe(
      "OpenAI",
    );
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
