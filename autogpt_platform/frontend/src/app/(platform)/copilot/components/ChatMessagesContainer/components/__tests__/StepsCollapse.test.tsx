import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { StepsCollapse } from "../StepsCollapse";

vi.mock("@phosphor-icons/react", () => ({
  ListBulletsIcon: () => <span data-testid="list-icon" />,
}));

vi.mock("@/components/molecules/Dialog/Dialog", async () => {
  const { createContext, useContext, useState } = await import("react");

  const Ctx = createContext<{
    open: boolean;
    setOpen: (v: boolean) => void;
  }>({ open: false, setOpen: () => {} });

  function Dialog({ children }: { children: React.ReactNode }) {
    const [open, setOpen] = useState(false);
    return <Ctx.Provider value={{ open, setOpen }}>{children}</Ctx.Provider>;
  }
  Dialog.Trigger = function Trigger({
    children,
  }: {
    children: React.ReactNode;
  }) {
    const ctx = useContext(Ctx);
    return (
      <span data-testid="dialog-trigger" onClick={() => ctx.setOpen(true)}>
        {children}
      </span>
    );
  };
  Dialog.Content = function Content({
    children,
  }: {
    children: React.ReactNode;
  }) {
    const ctx = useContext(Ctx);
    if (!ctx.open) return null;
    return <div data-testid="dialog-content">{children}</div>;
  };
  return { Dialog };
});

describe("StepsCollapse", () => {
  afterEach(() => {
    cleanup();
  });

  it("renders the 'Show steps' trigger button without opening the dialog", () => {
    render(
      <StepsCollapse>
        <div data-testid="step-body">step content</div>
      </StepsCollapse>,
    );

    expect(screen.getByText("Show steps")).toBeDefined();
    expect(screen.queryByTestId("dialog-content")).toBeNull();
  });

  it("opens the dialog and renders children on trigger click", () => {
    render(
      <StepsCollapse>
        <div data-testid="step-body">step content</div>
      </StepsCollapse>,
    );

    fireEvent.click(screen.getByTestId("dialog-trigger"));

    expect(screen.getByTestId("dialog-content")).toBeDefined();
    expect(screen.getByTestId("step-body")).toBeDefined();
    expect(screen.getByText("step content")).toBeDefined();
  });
});
