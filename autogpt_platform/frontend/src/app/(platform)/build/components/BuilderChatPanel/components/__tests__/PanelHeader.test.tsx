import { describe, expect, it, vi } from "vitest";

import { fireEvent, render, screen } from "@/tests/integrations/test-utils";

import { PanelHeader } from "../PanelHeader";

describe("PanelHeader", () => {
  const baseProps = {
    onClose: () => {},
    canRevert: false,
    revertTargetVersion: null as number | null,
    onRevert: () => {},
  };

  it("renders the panel title and the close button", () => {
    render(<PanelHeader {...baseProps} />);
    expect(screen.getByText("Chat with Builder")).toBeDefined();
    expect(screen.getByRole("button", { name: /close/i })).toBeDefined();
  });

  it("omits the revert button when revert is not available", () => {
    render(<PanelHeader {...baseProps} />);
    expect(screen.queryByRole("button", { name: /revert/i })).toBeNull();
  });

  it("shows the revert button with a generic aria-label when no target version is known", () => {
    render(<PanelHeader {...baseProps} canRevert revertTargetVersion={null} />);
    expect(
      screen.getByRole("button", { name: "Revert to previous version" }),
    ).toBeDefined();
  });

  it("shows the revert button with a version-specific aria-label when a target version is known", () => {
    render(<PanelHeader {...baseProps} canRevert revertTargetVersion={7} />);
    expect(
      screen.getByRole("button", { name: "Revert to version 7" }),
    ).toBeDefined();
  });

  it("invokes the revert and close callbacks on click", () => {
    const onClose = vi.fn();
    const onRevert = vi.fn();
    render(
      <PanelHeader
        {...baseProps}
        canRevert
        onClose={onClose}
        onRevert={onRevert}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /revert/i }));
    fireEvent.click(screen.getByRole("button", { name: /close/i }));
    expect(onRevert).toHaveBeenCalledTimes(1);
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
