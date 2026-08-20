import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, test, vi } from "vitest";

const { undo } = vi.hoisted(() => ({ undo: vi.fn() }));

vi.mock("../../../stores/historyStore", () => ({
  useHistoryStore: () => ({
    undo,
    redo: vi.fn(),
    canUndo: () => true,
    canRedo: () => true,
  }),
}));

import { UndoRedoButtons } from "../UndoRedoButtons";

describe("UndoRedoButtons", () => {
  test("leaves undo shortcuts to editable controls", () => {
    render(
      <>
        <UndoRedoButtons />
        <textarea aria-label="Expanded input" />
      </>,
    );

    fireEvent.keyDown(screen.getByLabelText("Expanded input"), {
      key: "z",
      ctrlKey: true,
    });
    expect(undo).not.toHaveBeenCalled();

    fireEvent.keyDown(window, { key: "z", ctrlKey: true });
    expect(undo).toHaveBeenCalledOnce();
  });
});
