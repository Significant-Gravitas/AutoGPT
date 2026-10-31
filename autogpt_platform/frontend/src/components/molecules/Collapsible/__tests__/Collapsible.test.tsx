import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { Collapsible } from "../Collapsible";

describe("Collapsible", () => {
  it("opens and closes from its trigger", async () => {
    render(<Collapsible trigger="Details">Hidden content</Collapsible>);
    const trigger = screen.getByRole("button", { name: /Details/ });

    expect(trigger.getAttribute("aria-expanded")).toBe("false");
    expect(screen.queryByText("Hidden content")).toBeNull();

    await userEvent.click(trigger);
    expect(trigger.getAttribute("aria-expanded")).toBe("true");
    expect(screen.getByText("Hidden content")).toBeDefined();

    await userEvent.click(trigger);
    expect(trigger.getAttribute("aria-expanded")).toBe("false");
  });

  it("opens by keyboard", async () => {
    render(<Collapsible trigger="Details">Hidden content</Collapsible>);
    await userEvent.tab();
    await userEvent.keyboard("{Enter}");
    expect(screen.getByText("Hidden content")).toBeDefined();
  });

  it("starts open with defaultOpen", () => {
    render(
      <Collapsible trigger="Details" defaultOpen>
        Shown content
      </Collapsible>,
    );
    expect(screen.getByText("Shown content")).toBeDefined();
  });

  it("follows a controlled open prop and reports toggles", async () => {
    const onOpenChange = vi.fn();
    const { rerender } = render(
      <Collapsible trigger="Details" open={false} onOpenChange={onOpenChange}>
        Controlled content
      </Collapsible>,
    );
    await userEvent.click(screen.getByRole("button", { name: /Details/ }));
    expect(onOpenChange).toHaveBeenCalledWith(true);
    expect(screen.queryByText("Controlled content")).toBeNull();

    rerender(
      <Collapsible trigger="Details" open onOpenChange={onOpenChange}>
        Controlled content
      </Collapsible>,
    );
    expect(screen.getByText("Controlled content")).toBeDefined();
  });
});
