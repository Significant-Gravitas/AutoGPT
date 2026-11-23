import { useState } from "react";
import { describe, expect, it } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { expectTypingKept } from "../typing";

interface Props {
  rewrite?: (text: string) => string;
}

function NumberInput({ rewrite = (text) => text }: Props) {
  const [value, setValue] = useState("");
  return (
    <input
      type="number"
      aria-label="Amount"
      value={value}
      onChange={(e) => setValue(rewrite(e.target.value))}
    />
  );
}

describe("expectTypingKept on a number input", () => {
  it("passes a field that keeps every keystroke", async () => {
    render(<NumberInput />);

    await expectTypingKept(
      screen.getByRole("spinbutton") as HTMLInputElement,
      "-9007199254740993",
    );
  });

  it("fails a field that rewrites what was typed so far, even if the end matches", async () => {
    render(<NumberInput rewrite={(text) => (text === "12" ? "1" : text)} />);

    await expect(
      expectTypingKept(
        screen.getByRole("spinbutton") as HTMLInputElement,
        "123",
      ),
    ).rejects.toThrow('after typing "12"');
  });
});
