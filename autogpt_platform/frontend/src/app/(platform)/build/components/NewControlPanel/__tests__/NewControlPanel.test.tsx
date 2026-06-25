import { render, screen } from "@testing-library/react";
import { describe, expect, test, vi } from "vitest";

vi.mock("../NewBlockMenu/BlockMenu/BlockMenu", () => ({
  BlockMenu: () => <div data-testid="block-menu" />,
}));

vi.mock("../NewSaveControl/NewSaveControl", () => ({
  NewSaveControl: () => <div data-testid="save-control" />,
}));

vi.mock("../NewSearchGraph/GraphMenu/GraphMenu", () => ({
  GraphSearchMenu: () => <div data-testid="graph-search" />,
}));

vi.mock("../UndoRedoButtons", () => ({
  UndoRedoButtons: () => <div data-testid="undo-redo" />,
}));

vi.mock("../useGraphSearchShortcut", () => ({
  useGraphSearchShortcut: vi.fn(),
}));

import { NewControlPanel } from "../NewControlPanel";

describe("NewControlPanel", () => {
  test("renders all controls when the graph is editable", () => {
    render(<NewControlPanel />);

    expect(screen.queryByTestId("block-menu")).not.toBeNull();
    expect(screen.queryByTestId("save-control")).not.toBeNull();
    expect(screen.queryByTestId("undo-redo")).not.toBeNull();
    expect(screen.queryByTestId("graph-search")).not.toBeNull();
  });

  test("renders only the graph search when read-only", () => {
    render(<NewControlPanel isReadOnly />);

    expect(screen.queryByTestId("graph-search")).not.toBeNull();
    expect(screen.queryByTestId("block-menu")).toBeNull();
    expect(screen.queryByTestId("save-control")).toBeNull();
    expect(screen.queryByTestId("undo-redo")).toBeNull();
  });
});
