import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { TooltipProvider } from "@radix-ui/react-tooltip";
import { DraftRecoveryPopup } from "../components/DraftRecoveryDialog/DraftRecoveryPopup";

const mockOnLoad = vi.fn();
const mockOnDiscard = vi.fn();

vi.mock("../components/DraftRecoveryDialog/useDraftRecoveryPopup", () => ({
  useDraftRecoveryPopup: vi.fn(() => ({
    isOpen: true,
    popupRef: { current: null },
    nodeCount: 3,
    edgeCount: 2,
    diff: {
      nodes: { added: 1, removed: 0, modified: 2 },
      edges: { added: 1, removed: 1, modified: 0 },
    },
    savedAt: Date.now(),
    onLoad: mockOnLoad,
    onDiscard: mockOnDiscard,
  })),
}));

vi.mock("framer-motion", () => ({
  AnimatePresence: ({ children }: { children: React.ReactNode }) => (
    <>{children}</>
  ),
  motion: {
    div: React.forwardRef(function MotionDiv(
      props: Record<string, unknown>,
      ref: React.Ref<HTMLDivElement>,
    ) {
      const {
        children,
        initial: _initial,
        animate: _animate,
        exit: _exit,
        transition: _transition,
        ...rest
      } = props as {
        children?: React.ReactNode;
        initial?: unknown;
        animate?: unknown;
        exit?: unknown;
        transition?: unknown;
        [key: string]: unknown;
      };
      return (
        <div ref={ref} {...rest}>
          {children}
        </div>
      );
    }),
  },
}));

function renderWithProviders(ui: React.ReactElement) {
  return render(<TooltipProvider>{ui}</TooltipProvider>);
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("DraftRecoveryPopup", () => {
  describe("when open with diff data", () => {
    it("shows the unsaved changes message", () => {
      renderWithProviders(<DraftRecoveryPopup isInitialLoadComplete={true} />);
      expect(screen.getByText("Unsaved changes found")).toBeDefined();
    });

    it("displays diff summary", () => {
      renderWithProviders(<DraftRecoveryPopup isInitialLoadComplete={true} />);
      const text = document.body.textContent;
      expect(text).toContain("+1/~2 blocks");
      expect(text).toContain("+1/-1 connections");
    });

    it("renders restore and discard buttons", () => {
      renderWithProviders(<DraftRecoveryPopup isInitialLoadComplete={true} />);
      expect(screen.getAllByText("Restore changes").length).toBeGreaterThan(0);
      expect(screen.getAllByText("Discard changes").length).toBeGreaterThan(0);
    });

    it("calls onLoad when restore is clicked", () => {
      renderWithProviders(<DraftRecoveryPopup isInitialLoadComplete={true} />);
      const buttons = screen.getAllByRole("button", {
        name: /restore changes/i,
      });
      fireEvent.click(buttons[0]);
      expect(mockOnLoad).toHaveBeenCalledOnce();
    });

    it("calls onDiscard when discard is clicked", () => {
      renderWithProviders(<DraftRecoveryPopup isInitialLoadComplete={true} />);
      const buttons = screen.getAllByRole("button", {
        name: /discard changes/i,
      });
      fireEvent.click(buttons[0]);
      expect(mockOnDiscard).toHaveBeenCalledOnce();
    });
  });

  describe("when closed", () => {
    it("renders nothing when isOpen is false", async () => {
      const { useDraftRecoveryPopup } = await import(
        "../components/DraftRecoveryDialog/useDraftRecoveryPopup"
      );
      vi.mocked(useDraftRecoveryPopup).mockReturnValue({
        isOpen: false,
        popupRef: { current: null },
        nodeCount: 0,
        edgeCount: 0,
        diff: null,
        savedAt: 0,
        onLoad: vi.fn(),
        onDiscard: vi.fn(),
      });

      const { container } = renderWithProviders(
        <DraftRecoveryPopup isInitialLoadComplete={true} />,
      );
      expect(container.textContent).toBe("");
    });
  });

  describe("when diff is null", () => {
    it("falls back to node/edge count display", async () => {
      const { useDraftRecoveryPopup } = await import(
        "../components/DraftRecoveryDialog/useDraftRecoveryPopup"
      );
      vi.mocked(useDraftRecoveryPopup).mockReturnValue({
        isOpen: true,
        popupRef: { current: null },
        nodeCount: 5,
        edgeCount: 1,
        diff: null,
        savedAt: Date.now(),
        onLoad: vi.fn(),
        onDiscard: vi.fn(),
      });

      renderWithProviders(<DraftRecoveryPopup isInitialLoadComplete={true} />);
      const text = document.body.textContent;
      expect(text).toContain("5 blocks");
      expect(text).toContain("1 connection");
    });

    it("uses singular for 1 block", async () => {
      const { useDraftRecoveryPopup } = await import(
        "../components/DraftRecoveryDialog/useDraftRecoveryPopup"
      );
      vi.mocked(useDraftRecoveryPopup).mockReturnValue({
        isOpen: true,
        popupRef: { current: null },
        nodeCount: 1,
        edgeCount: 0,
        diff: null,
        savedAt: Date.now(),
        onLoad: vi.fn(),
        onDiscard: vi.fn(),
      });

      renderWithProviders(<DraftRecoveryPopup isInitialLoadComplete={true} />);
      const text = document.body.textContent;
      expect(text).toContain("1 block,");
      expect(text).toContain("0 connections");
    });
  });
});
