import { describe, expect, test, vi } from "vitest";

import { fireEvent, render, screen } from "@/tests/integrations/test-utils";

import { IntegrationsHeader } from "../IntegrationsHeader";

describe("IntegrationsHeader", () => {
  test("renders the title and intro copy", () => {
    render(<IntegrationsHeader onConnect={() => {}} />);
    expect(
      screen.getByRole("heading", { name: /third party integrations/i }),
    ).toBeDefined();
    expect(screen.getByText(/Manage the 3rd party accounts/i)).toBeDefined();
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
