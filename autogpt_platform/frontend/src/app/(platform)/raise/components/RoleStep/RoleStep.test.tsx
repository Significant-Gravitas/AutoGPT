import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { RoleStep } from "./RoleStep";

describe("RoleStep", () => {
  test("does not show Something else as a preset pill", () => {
    render(<RoleStep selectedRole={null} color={null} onPick={vi.fn()} />);

    expect(screen.queryByRole("button", { name: "Something else" })).toBeNull();
    expect(screen.getByPlaceholderText("Type a role…")).toBeDefined();
  });

  test("submits trimmed custom role text", async () => {
    const onPick = vi.fn();
    render(<RoleStep selectedRole={null} color={null} onPick={onPick} />);

    await userEvent.type(
      screen.getByPlaceholderText("Type a role…"),
      "  UX Designer  ",
    );
    await userEvent.click(screen.getByRole("button", { name: "Add role" }));

    expect(onPick).toHaveBeenCalledWith("UX Designer");
  });

  test("collapses to the selected custom role", () => {
    render(
      <RoleStep selectedRole="UX Designer" color={null} onPick={vi.fn()} />,
    );

    expect(screen.getByRole("button", { name: "UX Designer" })).toBeDefined();
    expect(screen.queryByPlaceholderText("Type a role…")).toBeNull();
  });
});
