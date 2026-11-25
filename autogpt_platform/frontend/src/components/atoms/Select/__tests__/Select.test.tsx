import { StrictMode, useState } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Select } from "../Select";

const options = [
  { value: "alpha", label: "Alpha" },
  { value: "beta", label: "Beta" },
  { value: "disabled", label: "Disabled option", disabled: true },
];
const field = { id: "choice", label: "Choice", options };

beforeEach(() => {
  vi.spyOn(console, "warn").mockImplementation(() => {});
  vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
});

function expectNoWarnings() {
  expect(console.warn).not.toHaveBeenCalled();
  expect(console.error).not.toHaveBeenCalled();
}

describe("Select value lifecycle with Radix", () => {
  it.each([undefined, null, ""])(
    "selects and clears an explicitly supplied %s value without changing control mode",
    (emptyValue) => {
      const value = emptyValue as unknown as string | undefined;
      const onValueChange = vi.fn();
      const props = { ...field, placeholder: "Pick a choice", onValueChange };
      const { rerender } = render(<Select {...props} value={value} />, {
        wrapper: StrictMode,
      });
      const trigger = screen.getByRole("combobox", { name: "Choice" });
      expect(trigger.textContent).toBe("Pick a choice");
      expect(trigger.hasAttribute("data-placeholder")).toBe(true);

      rerender(<Select {...props} value="beta" />);
      expect(trigger.textContent).toBe("Beta");
      expect(trigger.hasAttribute("data-placeholder")).toBe(false);

      rerender(<Select {...props} value={value} />);
      expect(trigger.textContent).toBe("Pick a choice");
      expect(trigger.hasAttribute("data-placeholder")).toBe(true);
      expect(onValueChange).not.toHaveBeenCalled();
      expectNoWarnings();
    },
  );

  it("updates controlled selection through Radix and clears back to the label", async () => {
    const user = userEvent.setup();
    const onValueChange = vi.fn();
    function ControlledField() {
      const [value, setValue] = useState<string>();
      return (
        <>
          <Select
            {...field}
            value={value}
            onValueChange={(nextValue) => {
              setValue(nextValue);
              onValueChange(nextValue);
            }}
          />
          <button type="button" onClick={() => setValue(undefined)}>
            Clear
          </button>
        </>
      );
    }
    render(<ControlledField />);
    const trigger = screen.getByRole("combobox", { name: "Choice" });
    await user.click(trigger);
    await user.click(await screen.findByRole("option", { name: "Alpha" }));
    expect(trigger.textContent).toBe("Alpha");
    expect(onValueChange).toHaveBeenCalledExactlyOnceWith("alpha");

    await user.click(screen.getByRole("button", { name: "Clear" }));
    expect(trigger.textContent).toBe("Choice");
    expect(trigger.hasAttribute("data-placeholder")).toBe(true);

    await user.click(trigger);
    await user.click(await screen.findByRole("option", { name: "Beta" }));
    expect(trigger.textContent).toBe("Beta");
    expect(onValueChange.mock.calls).toEqual([["alpha"], ["beta"]]);
    expectNoWarnings();
  });

  it("keeps selection uncontrolled when value is omitted", async () => {
    const user = userEvent.setup();
    const onValueChange = vi.fn();
    const { rerender } = render(
      <Select {...field} onValueChange={onValueChange} />,
    );
    const trigger = screen.getByRole("combobox", { name: "Choice" });
    await user.click(trigger);
    await user.click(await screen.findByRole("option", { name: "Alpha" }));
    expect(trigger.textContent).toBe("Alpha");
    rerender(
      <Select {...field} onValueChange={onValueChange} hint="Updated hint" />,
    );
    expect(trigger.textContent).toBe("Alpha");
    await user.click(trigger);
    await user.click(await screen.findByRole("option", { name: "Beta" }));
    expect(trigger.textContent).toBe("Beta");
    expect(onValueChange.mock.calls).toEqual([["alpha"], ["beta"]]);
    expectNoWarnings();
  });

  it("allows uncontrolled selection without a change handler", async () => {
    const user = userEvent.setup();
    render(<Select {...field} />);
    const trigger = screen.getByRole("combobox", { name: "Choice" });
    await user.click(trigger);
    await user.click(await screen.findByRole("option", { name: "Beta" }));
    expect(trigger.textContent).toBe("Beta");
    expectNoWarnings();
  });

  it("respects disabled fields and options", async () => {
    const user = userEvent.setup();
    const onValueChange = vi.fn();
    const { rerender } = render(
      <Select
        {...field}
        value={undefined}
        onValueChange={onValueChange}
        disabled
      />,
    );
    await user.click(screen.getByRole("combobox", { name: "Choice" }));
    expect(screen.queryByRole("listbox")).toBeNull();
    rerender(
      <Select {...field} value={undefined} onValueChange={onValueChange} />,
    );
    await user.click(screen.getByRole("combobox", { name: "Choice" }));
    const disabledOption = await screen.findByRole("option", {
      name: "Disabled option",
    });
    expect(disabledOption.getAttribute("aria-disabled")).toBe("true");
    await user.click(disabledOption);
    expect(onValueChange).not.toHaveBeenCalled();
    expectNoWarnings();
  });

  it.each(["__proto__", '<img src=x onerror="alert(1)">'])(
    "preserves the literal option value %s",
    async (value) => {
      const user = userEvent.setup();
      const onValueChange = vi.fn();
      const props = {
        ...field,
        options: [{ value, label: value }],
        onValueChange,
      };
      const { rerender } = render(<Select {...props} value={undefined} />);
      const trigger = screen.getByRole("combobox", { name: "Choice" });
      await user.click(trigger);
      await user.click(await screen.findByRole("option", { name: value }));
      expect(onValueChange).toHaveBeenCalledExactlyOnceWith(value);
      rerender(<Select {...props} value={value} />);
      expect(trigger.textContent).toBe(value);
      expect(document.querySelector("img")).toBeNull();
      expectNoWarnings();
    },
  );
});
