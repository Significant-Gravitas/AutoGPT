import { renderHook } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { useAnotherComputer } from "../useAnotherComputer";
import { LAPTOP, STUDIO, WORKSTATION } from "./computers";

describe("which computers the notice names", () => {
  test("one that joins in the very render another is chosen keeps its notice", () => {
    const { result, rerender } = renderHook(useAnotherComputer, {
      initialProps: {
        isOpen: true,
        machines: [WORKSTATION],
        selectedMachineID: WORKSTATION.machine_id,
      },
    });
    rerender({
      isOpen: true,
      machines: [WORKSTATION, LAPTOP],
      selectedMachineID: WORKSTATION.machine_id,
    });
    expect(result.current.connectedNotice).toBe(
      "Laptop · macOS connected. Choose it under Computer.",
    );

    // The list's answer and the person's choice land together.
    rerender({
      isOpen: true,
      machines: [WORKSTATION, LAPTOP, STUDIO],
      selectedMachineID: LAPTOP.machine_id,
    });

    expect(result.current.connectedNotice).toBe(
      "Studio · macOS connected. Choose it under Computer.",
    );
  });
});
