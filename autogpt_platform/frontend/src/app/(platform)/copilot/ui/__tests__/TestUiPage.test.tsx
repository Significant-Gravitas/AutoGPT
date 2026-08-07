import {
  cleanup,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it } from "vitest";
import { TestUiPage } from "../TestUiPage";

describe("TestUiPage", () => {
  afterEach(cleanup);

  it("renders the complete tool catalog and toggles raw samples", () => {
    render(<TestUiPage />);

    expect(
      screen.getByRole("heading", { name: "Tool UI — full catalog" }),
    ).toBeDefined();
    expect(screen.getByRole("heading", { name: "Agents" })).toBeDefined();
    expect(
      screen.getByRole("heading", { name: "Web & browser" }),
    ).toBeDefined();
    expect(
      screen.getByText("ask_question — user answers inline"),
    ).toBeDefined();

    fireEvent.click(screen.getByRole("button", { name: "Show raw data" }));

    expect(screen.getByRole("button", { name: "Hide raw data" })).toBeDefined();
    expect(screen.getAllByText(/"tool":/).length).toBeGreaterThan(10);
  }, 15_000);
});
