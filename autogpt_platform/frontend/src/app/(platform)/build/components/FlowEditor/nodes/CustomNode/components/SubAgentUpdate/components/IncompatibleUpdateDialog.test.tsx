import { render, screen } from "@testing-library/react";
import { describe, expect, test, vi } from "vitest";
import { IncompatibleUpdateDialog } from "./IncompatibleUpdateDialog";

describe("IncompatibleUpdateDialog", () => {
  test("lists the inputs that would break and the types that changed", () => {
    render(
      <IncompatibleUpdateDialog
        isOpen
        onClose={vi.fn()}
        onConfirm={vi.fn()}
        currentVersion={1}
        latestVersion={2}
        agentName="writer"
        incompatibilities={{
          missingInputs: ["topic"],
          missingOutputs: [],
          newInputs: ["tone"],
          newOutputs: [],
          newRequiredInputs: ["audience"],
          inputTypeMismatches: [
            { name: "count", oldType: "string", newType: "number" },
          ],
        }}
      />,
    );

    expect(screen.getByText("Input Changes")).toBeDefined();
    expect(screen.getByText("topic")).toBeDefined();
    expect(screen.getByText("tone")).toBeDefined();
    expect(screen.getByText("Type Changed")).toBeDefined();
    expect(screen.getByText("count (string → number)")).toBeDefined();
    expect(screen.getByText("New Required Inputs")).toBeDefined();
    expect(screen.getByText("audience")).toBeDefined();
  });
});
