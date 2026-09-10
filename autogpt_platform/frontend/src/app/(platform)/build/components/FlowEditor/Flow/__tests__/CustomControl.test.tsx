import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { fireEvent, render } from "@testing-library/react";
import { ReactElement } from "react";
import { describe, expect, test, vi } from "vitest";

vi.mock("@xyflow/react", () => ({
  useReactFlow: () => ({
    zoomIn: vi.fn(),
    zoomOut: vi.fn(),
    fitView: vi.fn(),
  }),
}));

vi.mock("@/app/(platform)/build/stores/tutorialStore", () => ({
  useTutorialStore: () => ({
    isTutorialRunning: false,
    setIsTutorialRunning: vi.fn(),
  }),
}));

vi.mock("../../tutorial", () => ({
  startTutorial: vi.fn(),
  setTutorialLoadingCallback: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ push: vi.fn() }),
}));

import { CustomControls } from "../components/CustomControl";

function renderControls(ui: ReactElement) {
  return render(<TooltipProvider>{ui}</TooltipProvider>);
}

describe("CustomControls", () => {
  test("lock toggle is enabled and works when not read-only", () => {
    const setIsLocked = vi.fn();
    const { container } = renderControls(
      <CustomControls isLocked={false} setIsLocked={setIsLocked} />,
    );

    const lockButton = container.querySelector<HTMLButtonElement>(
      '[data-id="lock-button"]',
    );
    expect(lockButton).not.toBeNull();
    expect(lockButton!.disabled).toBe(false);

    fireEvent.click(lockButton!);
    expect(setIsLocked).toHaveBeenCalledWith(true);
  });

  test("lock toggle is disabled when read-only", () => {
    const setIsLocked = vi.fn();
    const { container } = renderControls(
      <CustomControls isLocked={true} setIsLocked={setIsLocked} isReadOnly />,
    );

    const lockButton = container.querySelector<HTMLButtonElement>(
      '[data-id="lock-button"]',
    );
    expect(lockButton).not.toBeNull();
    expect(lockButton!.disabled).toBe(true);
    expect(lockButton!.textContent).toContain(
      "Canvas is locked because this is a read-only graph",
    );

    fireEvent.click(lockButton!);
    expect(setIsLocked).not.toHaveBeenCalled();
  });
});
