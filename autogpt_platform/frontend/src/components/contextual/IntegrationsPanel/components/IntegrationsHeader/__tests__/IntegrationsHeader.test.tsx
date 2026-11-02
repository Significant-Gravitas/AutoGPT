import { describe, expect, test, vi } from "vitest";

import { fireEvent, render, screen } from "@/tests/integrations/test-utils";

import { IntegrationsHeader } from "../IntegrationsHeader";

describe("IntegrationsHeader", () => {
  test("names both kinds of thing the page now holds", () => {
    // The page led with third-party tool connectors, which stopped being the
    // whole story once AI subscriptions got their own section above them --
    // and reads as wrong for a self-hosted connection, which is not a third
    // party at all.
    render(<IntegrationsHeader onConnect={() => {}} />);
    expect(
      screen.getByRole("heading", { name: /^integrations$/i }),
    ).toBeDefined();
    expect(screen.getByText(/Connect AI subscriptions/i)).toBeDefined();
    expect(
      screen.getByText(/third-party tools for them to use/i),
    ).toBeDefined();
  });

  test("invokes onConnect when any 'Connect Service' button is clicked", () => {
    const onConnect = vi.fn();
    render(<IntegrationsHeader onConnect={onConnect} />);
    // There are two responsive copies of the button; clicking either should
    // trigger the same callback. We click the first one we find.
    const buttons = screen.getAllByRole("button", { name: /connect service/i });
    expect(buttons.length).toBeGreaterThan(0);
    fireEvent.click(buttons[0]);
    expect(onConnect).toHaveBeenCalledTimes(1);
  });
});
