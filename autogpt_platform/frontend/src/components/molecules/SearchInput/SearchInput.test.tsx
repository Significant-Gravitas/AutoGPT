import { useState } from "react";
import { describe, expect, test, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { render, screen } from "@/tests/integrations/test-utils";
import { SearchInput } from "./SearchInput";

function Harness({
  initial = "",
  onChange,
  ...rest
}: {
  initial?: string;
  onChange?: (next: string) => void;
} & Omit<React.ComponentProps<typeof SearchInput>, "value" | "onChange">) {
  const [value, setValue] = useState(initial);
  return (
    <SearchInput
      {...rest}
      value={value}
      onChange={(next) => {
        setValue(next);
        onChange?.(next);
      }}
    />
  );
}

describe("SearchInput", () => {
  test("renders placeholder text on the input", () => {
    render(<Harness placeholder="Find creators" />);
    expect(screen.getByPlaceholderText("Find creators")).toBeDefined();
  });

  test("uses placeholder as the accessible label by default", () => {
    render(<Harness placeholder="Find creators" />);
    expect(
      screen.getByRole("searchbox", { name: "Find creators" }),
    ).toBeDefined();
  });

  test("prefers explicit aria-label over placeholder", () => {
    render(
      <Harness placeholder="Find creators" aria-label="Creator search input" />,
    );
    expect(
      screen.getByRole("searchbox", { name: "Creator search input" }),
    ).toBeDefined();
  });

  test("fires onChange for every keystroke", async () => {
    const user = userEvent.setup();
    const handleChange = vi.fn();
    render(<Harness placeholder="Search" onChange={handleChange} />);

    await user.type(screen.getByRole("searchbox"), "abc");

    expect(handleChange).toHaveBeenCalledTimes(3);
    expect(handleChange).toHaveBeenNthCalledWith(1, "a");
    expect(handleChange).toHaveBeenNthCalledWith(2, "ab");
    expect(handleChange).toHaveBeenNthCalledWith(3, "abc");
  });

  test("clear button resets the value and emits empty string", async () => {
    const user = userEvent.setup();
    const handleChange = vi.fn();
    render(
      <Harness placeholder="Search" initial="agent" onChange={handleChange} />,
    );

    const clearButton = screen.getByRole("button", { name: "Clear search" });
    await user.click(clearButton);

    expect(handleChange).toHaveBeenLastCalledWith("");
    expect((screen.getByRole("searchbox") as HTMLInputElement).value).toBe("");
  });

  test("hides the clear button when input is empty", () => {
    render(<Harness placeholder="Search" />);
    expect(screen.queryByRole("button", { name: "Clear search" })).toBeNull();
  });

  test("hides the clear button when disabled, even with a value", () => {
    render(<Harness placeholder="Search" initial="agent" disabled />);
    expect(screen.queryByRole("button", { name: "Clear search" })).toBeNull();
  });

  test("shows a loading status indicator instead of the clear button while loading", () => {
    render(<Harness placeholder="Search" initial="agent" loading />);
    expect(screen.getByRole("status", { name: "Searching" })).toBeDefined();
    expect(screen.queryByRole("button", { name: "Clear search" })).toBeNull();
  });
});
