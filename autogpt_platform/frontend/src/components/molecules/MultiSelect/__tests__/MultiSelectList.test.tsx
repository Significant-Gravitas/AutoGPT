import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import {
  MultiSelector,
  MultiSelectorContent,
  MultiSelectorInput,
  MultiSelectorItem,
  MultiSelectorList,
  MultiSelectorTrigger,
} from "../MultiSelect";

const OPTIONS = ["Apples", "Bananas", "Cherries"];

function Picker({
  initial = ["Apples"],
  onValuesChange = vi.fn(),
  disabled,
}: {
  initial?: string[];
  onValuesChange?: (value: string[]) => void;
  disabled?: string;
}) {
  const [values, setValues] = useState(initial);
  return (
    <MultiSelector
      values={values}
      onValuesChange={(next) => {
        onValuesChange(next);
        setValues(next);
      }}
    >
      <MultiSelectorTrigger>
        <MultiSelectorInput aria-label="Fruit" placeholder="Pick fruit" />
      </MultiSelectorTrigger>
      <MultiSelectorContent>
        <MultiSelectorList>
          {OPTIONS.map((option) => (
            <MultiSelectorItem
              key={option}
              value={option}
              disabled={option === disabled}
            >
              {option}
            </MultiSelectorItem>
          ))}
        </MultiSelectorList>
      </MultiSelectorContent>
    </MultiSelector>
  );
}

describe("MultiSelector list", () => {
  it("opens its list on focus and closes it on blur", () => {
    render(<Picker />);
    const input = screen.getByLabelText("Fruit");

    expect(screen.queryByRole("listbox")).toBeNull();
    fireEvent.focus(input);
    expect(screen.getByRole("listbox")).toBeDefined();
    expect(screen.getAllByRole("option")).toHaveLength(3);

    fireEvent.blur(input);
    expect(screen.queryByRole("listbox")).toBeNull();
  });

  it("adds a picked option and shows it as a chip", async () => {
    const onValuesChange = vi.fn();
    render(<Picker onValuesChange={onValuesChange} />);
    fireEvent.focus(screen.getByLabelText("Fruit"));

    await userEvent.click(screen.getByRole("option", { name: "Bananas" }));

    expect(onValuesChange).toHaveBeenLastCalledWith(["Apples", "Bananas"]);
    expect(
      screen.getByRole("button", { name: "Remove Bananas option" }),
    ).toBeDefined();
  });

  it("removes an option when it is picked again", async () => {
    const onValuesChange = vi.fn();
    render(<Picker onValuesChange={onValuesChange} />);
    fireEvent.focus(screen.getByLabelText("Fruit"));

    await userEvent.click(screen.getByRole("option", { name: "Apples" }));

    expect(onValuesChange).toHaveBeenLastCalledWith([]);
  });

  it("removes a chip with its remove button", async () => {
    const onValuesChange = vi.fn();
    render(
      <Picker
        initial={["Apples", "Cherries"]}
        onValuesChange={onValuesChange}
      />,
    );

    await userEvent.click(
      screen.getByRole("button", { name: "Remove Apples option" }),
    );

    expect(onValuesChange).toHaveBeenLastCalledWith(["Cherries"]);
    expect(screen.queryByText("Apples")).toBeNull();
  });

  it("says when nothing matches the filter", async () => {
    render(<Picker />);
    const input = screen.getByLabelText("Fruit");
    fireEvent.focus(input);

    await userEvent.type(input, "zzz");

    expect(screen.getByText("No results found")).toBeDefined();
    expect(screen.queryAllByRole("option")).toHaveLength(0);
  });

  it("marks a disabled option", () => {
    render(<Picker disabled="Cherries" />);
    fireEvent.focus(screen.getByLabelText("Fruit"));

    const option = screen.getByRole("option", { name: "Cherries" });
    expect(option.getAttribute("aria-disabled")).toBe("true");
    expect(option.className).toContain("cursor-not-allowed");
  });

  it("removes the highlighted chip with Backspace and clears the highlight with Escape", () => {
    const onValuesChange = vi.fn();
    render(
      <Picker
        initial={["Apples", "Bananas", "Cherries"]}
        onValuesChange={onValuesChange}
      />,
    );
    const input = screen.getByLabelText("Fruit");

    fireEvent.keyDown(input, { key: "ArrowLeft" });
    fireEvent.keyDown(input, { key: "ArrowLeft" });
    fireEvent.keyDown(input, { key: "Backspace" });
    expect(onValuesChange).toHaveBeenLastCalledWith(["Apples", "Cherries"]);

    fireEvent.keyDown(input, { key: "Escape" });
    fireEvent.keyDown(input, { key: "Backspace" });
    expect(onValuesChange).toHaveBeenLastCalledWith(["Apples"]);
  });
});
