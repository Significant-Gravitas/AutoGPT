import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";
import { ToolUiDebugPage } from "../ToolUiDebugPage";

describe("ToolUiDebugPage", () => {
  afterEach(cleanup);

  it("switches between the new and old tool UI variants", async () => {
    const user = userEvent.setup();
    render(<ToolUiDebugPage />);

    expect(
      screen.getByRole("heading", { name: "Tool UI debug" }),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: "new" }).getAttribute("aria-pressed"),
    ).toBe("true");

    await user.click(screen.getByRole("button", { name: "old" }));

    expect(
      screen.getByRole("button", { name: "old" }).getAttribute("aria-pressed"),
    ).toBe("true");
  });
});
