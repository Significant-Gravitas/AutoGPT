import { describe, expect, test, vi } from "vitest";

import {
  fireEvent,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";

import { IncompatibleUpdateDialog } from "../IncompatibleUpdateDialog";

const incompatibilities = {
  missingInputs: ["topic"],
  missingOutputs: [],
  newInputs: ["subject"],
  newOutputs: ["summary"],
  newRequiredInputs: ["subject"],
  inputTypeMismatches: [
    { name: "count", oldType: "string", newType: "number" },
  ],
};

function renderDialog() {
  const onClose = vi.fn();
  const onConfirm = vi.fn();
  render(
    <IncompatibleUpdateDialog
      isOpen
      onClose={onClose}
      onConfirm={onConfirm}
      currentVersion={1}
      latestVersion={2}
      agentName="research_helper"
      incompatibilities={incompatibilities}
    />,
  );
  return { onClose, onConfirm };
}

describe("IncompatibleUpdateDialog", () => {
  test("lists every removed, added, retyped and newly required item", async () => {
    renderDialog();
    const dialog = await screen.findByRole("dialog");

    expect(within(dialog).getByText(/from v1 to v2/)).toBeDefined();
    expect(within(dialog).getByText("Research Helper")).toBeDefined();

    const inputs = within(dialog).getByText("Input Changes").parentElement!;
    expect(within(inputs).getByText("topic").tagName).toBe("CODE");
    expect(within(inputs).getByText("subject").tagName).toBe("CODE");

    const outputs = within(dialog).getByText("Output Changes").parentElement!;
    expect(within(outputs).getByText("None")).toBeDefined();
    expect(within(outputs).getByText("summary")).toBeDefined();

    expect(within(dialog).getByText("count (string → number)").tagName).toBe(
      "CODE",
    );
    const required = within(dialog).getByText(
      "These inputs are now required:",
    ).parentElement!;
    expect(within(required).getByText("subject").tagName).toBe("CODE");
  });

  test("confirms and cancels through the footer buttons", async () => {
    const { onClose, onConfirm } = renderDialog();
    fireEvent.click(
      await screen.findByRole("button", { name: "Update Anyway" }),
    );
    expect(onConfirm).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onClose).toHaveBeenCalled();
  });
});
