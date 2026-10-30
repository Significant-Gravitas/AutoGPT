import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../components/SelectPrimitives";

it("preserves the primitive's uncontrolled defaultValue and subsequent selection", async () => {
  const user = userEvent.setup();
  const onValueChange = vi.fn();
  function UncontrolledSelect({ defaultValue }: { defaultValue: string }) {
    return (
      <Select defaultValue={defaultValue} onValueChange={onValueChange}>
        <SelectTrigger aria-label="Timezone">
          <SelectValue placeholder="Choose a timezone" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="utc">UTC</SelectItem>
          <SelectItem value="chicago">Chicago</SelectItem>
        </SelectContent>
      </Select>
    );
  }
  const { rerender } = render(<UncontrolledSelect defaultValue="utc" />);
  const trigger = screen.getByRole("combobox", { name: "Timezone" });
  expect(trigger.textContent).toBe("UTC");
  rerender(<UncontrolledSelect defaultValue="chicago" />);
  expect(trigger.textContent).toBe("UTC");
  await user.click(trigger);
  await user.click(await screen.findByRole("option", { name: "Chicago" }));
  expect(trigger.textContent).toBe("Chicago");
  rerender(<UncontrolledSelect defaultValue="utc" />);
  expect(trigger.textContent).toBe("Chicago");
  expect(onValueChange).toHaveBeenCalledExactlyOnceWith("chicago");
});
