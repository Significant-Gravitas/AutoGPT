import { createRef } from "react";
import { useForm } from "react-hook-form";
import { renderHook } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  cleanup,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { useControlPanelStore } from "../../../stores/controlPanelStore";
import { ControlPanelButton } from "../ControlPanelButton";
import { BlockMenu } from "../NewBlockMenu/BlockMenu/BlockMenu";
import { NewSaveControl } from "../NewSaveControl/NewSaveControl";
import { useNewSaveControl } from "../NewSaveControl/useNewSaveControl";
import { GraphSearchMenu } from "../NewSearchGraph/GraphMenu/GraphMenu";
import { UndoRedoButtons } from "../UndoRedoButtons";

vi.mock("../NewSaveControl/useNewSaveControl", () => ({
  useNewSaveControl: vi.fn(),
}));
vi.mock("../NewBlockMenu/BlockMenuContent/BlockMenuContent", () => ({
  BlockMenuContent: () => <div data-testid="block-menu-content" />,
}));
vi.mock("../NewSearchGraph/GraphMenuContent/GraphContent", () => ({
  GraphSearchContent: () => <div data-testid="graph-search-content" />,
}));
vi.mock("@xyflow/react", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@xyflow/react")>()),
  useReactFlow: () => ({ fitView: vi.fn() }),
}));

function mockSaveControl() {
  const { result } = renderHook(() =>
    useForm({ defaultValues: { name: "", description: "" } }),
  );
  vi.mocked(useNewSaveControl).mockReturnValue({
    form: result.current as never,
    isSaving: false,
    graphVersion: undefined,
    handleSave: vi.fn(),
  });
}

beforeEach(() => {
  useControlPanelStore.setState({
    blockMenuOpen: false,
    saveControlOpen: false,
    graphSearchOpen: false,
    forceOpenBlockMenu: false,
    forceOpenSave: false,
  });
  mockSaveControl();
});

afterEach(() => {
  cleanup();
});

describe("control panel buttons", () => {
  it("every toolbar button has an accessible name", () => {
    render(
      <>
        <BlockMenu />
        <GraphSearchMenu />
        <NewSaveControl />
        <UndoRedoButtons />
      </>,
    );

    for (const name of ["Blocks", "Search workflow", "Save", "Undo", "Redo"]) {
      expect(screen.getByRole("button", { name })).toBeDefined();
    }
  });

  it("Blocks is a single button, not a button inside a button", () => {
    render(<BlockMenu />);

    const blocks = screen.getByTestId("blocks-control-blocks-button");
    expect(blocks.tagName).toBe("BUTTON");
    expect(within(blocks).queryByRole("button")).toBeNull();
    expect(blocks.parentElement?.closest("button")).toBeNull();
  });

  it("Save opens with the keyboard and returns focus on Escape", async () => {
    const user = userEvent.setup();
    render(<NewSaveControl />);
    const save = screen.getByTestId("save-control-save-button");

    await user.tab();
    expect(document.activeElement).toBe(save);

    await user.keyboard("{Enter}");
    expect(await screen.findByTestId("save-control-name-input")).toBeDefined();

    await user.keyboard("{Escape}");
    await waitFor(() =>
      expect(screen.queryByTestId("save-control-name-input")).toBeNull(),
    );
    expect(document.activeElement).toBe(save);
  });

  it("Search workflow is reachable with Tab", async () => {
    const user = userEvent.setup();
    render(<GraphSearchMenu />);

    await user.tab();
    expect(document.activeElement).toBe(
      screen.getByTestId("graph-search-control-button"),
    );
  });

  it("forwards its ref to a real button", () => {
    const ref = createRef<HTMLButtonElement>();
    render(<ControlPanelButton ref={ref} aria-label="Test" />);

    expect(ref.current).toBeInstanceOf(HTMLButtonElement);
    expect(ref.current?.type).toBe("button");
  });
});
