import { Key } from "@/services/storage/local-storage";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";
import SettingsAgentPage from "../agent/page";

afterEach(() => {
  window.localStorage.clear();
});

describe("SettingsAgentPage", () => {
  it("defaults to Technical and persists the Compact choice", async () => {
    render(<SettingsAgentPage />);

    expect(
      screen.getByText("How do you want your agent to communicate?"),
    ).toBeDefined();
    const technical = screen.getByRole("radio", { name: /technical/i });
    const compact = screen.getByRole("radio", { name: /compact/i });
    expect(technical.getAttribute("aria-checked")).toBe("true");

    await userEvent.setup().click(compact);

    await waitFor(() =>
      expect(compact.getAttribute("aria-checked")).toBe("true"),
    );
    expect(window.localStorage.getItem(Key.COMMUNICATION_MODE)).toBe("compact");
  });
});
