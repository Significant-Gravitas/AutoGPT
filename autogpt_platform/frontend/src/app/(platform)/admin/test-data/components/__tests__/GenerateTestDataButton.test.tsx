import { render, screen, fireEvent } from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, test, vi } from "vitest";
import type { GenerateTestDataResponse } from "@/app/api/__generated__/models/generateTestDataResponse";

const hookState = vi.hoisted(() => ({
  isDialogOpen: false,
  scriptType: "e2e" as const,
  result: null as GenerateTestDataResponse | null,
  isPending: false,
  setScriptType: vi.fn(),
  openDialog: vi.fn(),
  closeDialog: vi.fn(),
  generate: vi.fn(),
}));

vi.mock("../useGenerateTestDataButton", () => ({
  useGenerateTestDataButton: () => hookState,
}));

import { GenerateTestDataButton } from "../GenerateTestDataButton";

beforeEach(() => {
  hookState.isDialogOpen = false;
  hookState.result = null;
  hookState.isPending = false;
  hookState.setScriptType.mockReset();
  hookState.openDialog.mockReset();
  hookState.closeDialog.mockReset();
  hookState.generate.mockReset();
});

describe("GenerateTestDataButton", () => {
  test("renders the trigger button and opens the dialog on click", () => {
    render(<GenerateTestDataButton />);
    const trigger = screen.getByRole("button", { name: "Generate Test Data" });
    fireEvent.click(trigger);
    expect(hookState.openDialog).toHaveBeenCalledTimes(1);
  });

  test("shows the script selector and warning when the dialog is open", () => {
    hookState.isDialogOpen = true;
    render(<GenerateTestDataButton />);
    expect(screen.getByText("Script Type")).toBeDefined();
    expect(
      screen.getByText(/only available in local environments/i),
    ).toBeDefined();
  });

  test("invokes generate from the dialog action", () => {
    hookState.isDialogOpen = true;
    render(<GenerateTestDataButton />);
    const actions = screen.getAllByRole("button", {
      name: "Generate Test Data",
    });
    fireEvent.click(actions[actions.length - 1]);
    expect(hookState.generate).toHaveBeenCalledTimes(1);
  });

  test("shows a loading label and disables actions while pending", () => {
    hookState.isDialogOpen = true;
    hookState.isPending = true;
    render(<GenerateTestDataButton />);
    const generating = screen.getByRole("button", { name: "Generating..." });
    expect(generating).toBeDefined();
    expect((generating as HTMLButtonElement).disabled).toBe(true);
  });

  test("renders a successful result with details", () => {
    hookState.isDialogOpen = true;
    hookState.result = {
      success: true,
      message: "E2E test data generated successfully",
      details: { users_created: 7 },
    };
    render(<GenerateTestDataButton />);
    expect(
      screen.getByText("E2E test data generated successfully"),
    ).toBeDefined();
    expect(screen.getByText(/users created/i)).toBeDefined();
  });
});
