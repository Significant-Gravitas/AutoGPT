import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, test, vi } from "vitest";
import { MultiSelect, type MultiSelectOption } from "./MultiSelect";

const OPTIONS: MultiSelectOption[] = [
  { value: "one", label: "One" },
  { value: "two", label: "Two" },
  { value: "three", label: "Three" },
];

function Harness({
  initial = [],
  onValueChange = vi.fn(),
}: {
  initial?: string[];
  onValueChange?: (value: string[]) => void;
}) {
  const [value, setValue] = useState(initial);
  return (
    <MultiSelect
      aria-label="Options"
      options={OPTIONS}
      value={value}
      onValueChange={(next) => {
        setValue(next);
        onValueChange(next);
      }}
    />
  );
}

describe("MultiSelect", () => {
  test("shows the selected options as chips, by label", () => {
    render(<Harness initial={["one", "two"]} />);

    expect(screen.getByText("One")).toBeDefined();
    expect(screen.getByText("Two")).toBeDefined();
    expect(screen.queryByText("Three")).toBeNull();
  });

  test("shows the placeholder only while nothing is selected", () => {
    const { unmount } = render(<Harness />);
    expect(screen.getByPlaceholderText("Select options...")).toBeDefined();
    unmount();

    render(<Harness initial={["one"]} />);
    expect(screen.queryByPlaceholderText("Select options...")).toBeNull();
  });

  test("adds an option picked from the list", async () => {
    const user = userEvent.setup();
    const onValueChange = vi.fn();
    render(<Harness initial={["one"]} onValueChange={onValueChange} />);

    await user.click(screen.getByRole("combobox", { name: "Options" }));
    await user.click(await screen.findByRole("option", { name: "Three" }));

    expect(onValueChange).toHaveBeenLastCalledWith(["one", "three"]);
  });

  test("removes an option through its chip", async () => {
    const user = userEvent.setup();
    const onValueChange = vi.fn();
    render(<Harness initial={["one", "two"]} onValueChange={onValueChange} />);

    const chip = screen.getByText("Two").closest("[data-slot='combobox-chip']");
    await user.click(chip!.querySelector("button")!);

    expect(onValueChange).toHaveBeenLastCalledWith(["one"]);
  });

  test("removes the last option with Backspace in the empty input", async () => {
    const user = userEvent.setup();
    const onValueChange = vi.fn();
    render(<Harness initial={["one", "two"]} onValueChange={onValueChange} />);

    await user.click(screen.getByRole("combobox", { name: "Options" }));
    await user.keyboard("{Backspace}");

    expect(onValueChange).toHaveBeenLastCalledWith(["one"]);
  });
});
