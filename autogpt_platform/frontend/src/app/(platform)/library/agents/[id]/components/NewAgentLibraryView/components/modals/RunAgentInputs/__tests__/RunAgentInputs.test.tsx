import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { expectTypingKept } from "@/tests/integrations/typing";
import type { BlockIOSubSchema } from "@/lib/autogpt-server-api/types";
import { RunAgentInputs } from "../RunAgentInputs";

const integerSchema = { title: "N", type: "integer" } as BlockIOSubSchema;

function ControlledInput({ onChange }: { onChange: (value: unknown) => void }) {
  const [value, setValue] = useState<unknown>();
  return (
    <RunAgentInputs
      schema={integerSchema}
      value={value}
      onChange={(next) => {
        setValue(next);
        onChange(next);
      }}
    />
  );
}

describe("RunAgentInputs integer input", () => {
  it("keeps a decimal as typed and sends it as a number", async () => {
    const onChange = vi.fn();
    render(<ControlledInput onChange={onChange} />);

    await expectTypingKept(
      screen.getByRole("spinbutton") as HTMLInputElement,
      "3.7",
    );

    expect(onChange).toHaveBeenLastCalledWith(3.7);
  });

  it("shows a stored digit string unchanged", () => {
    render(
      <RunAgentInputs
        schema={integerSchema}
        value="9007199254740993"
        onChange={vi.fn()}
      />,
    );
    expect(screen.getByRole("spinbutton")).toHaveProperty(
      "value",
      "9007199254740993",
    );
  });
});
