import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useCopilotModal } from "../../../../useCopilotModal";
import { ComposerPlusMenu } from "../ComposerPlusMenu";

afterEach(() => {
  vi.clearAllMocks();
});

function ModalProbe() {
  const { modal } = useCopilotModal();
  return <div data-testid="modal-probe">{modal ?? "none"}</div>;
}

function openMenu() {
  fireEvent.pointerDown(screen.getByTestId("composer-plus-button"), {
    button: 0,
  });
}

describe("ComposerPlusMenu", () => {
  it("lists the four actions in fixed order", async () => {
    render(<ComposerPlusMenu onFilesSelected={vi.fn()} />);
    openMenu();

    const items = await screen.findAllByRole("menuitem");
    expect(items.map((item) => item.textContent)).toEqual([
      "Attach file",
      "Integrations",
      "Skills",
      "Scheduled",
    ]);
  });

  it("selecting Skills opens the skills modal via query state", async () => {
    render(
      <>
        <ComposerPlusMenu onFilesSelected={vi.fn()} />
        <ModalProbe />
      </>,
    );
    openMenu();

    fireEvent.click(await screen.findByRole("menuitem", { name: /skills/i }));

    expect(screen.getByTestId("modal-probe").textContent).toBe("skills");
  });

  it("selecting Scheduled and Integrations open their modals", async () => {
    render(
      <>
        <ComposerPlusMenu onFilesSelected={vi.fn()} />
        <ModalProbe />
      </>,
    );
    openMenu();
    fireEvent.click(
      await screen.findByRole("menuitem", { name: /scheduled/i }),
    );
    expect(screen.getByTestId("modal-probe").textContent).toBe("scheduled");

    openMenu();
    fireEvent.click(
      await screen.findByRole("menuitem", { name: /integrations/i }),
    );
    expect(screen.getByTestId("modal-probe").textContent).toBe("integrations");
  });

  it("nests both upload sources under Attach file when the workspace option is enabled", async () => {
    const onUseWorkspaceFile = vi.fn();
    render(
      <ComposerPlusMenu
        onFilesSelected={vi.fn()}
        onUseWorkspaceFile={onUseWorkspaceFile}
        showWorkspaceOption={true}
      />,
    );
    openMenu();

    const attachTrigger = await screen.findByRole("menuitem", {
      name: /attach file/i,
    });
    fireEvent.click(attachTrigger);

    fireEvent.click(
      await screen.findByRole("menuitem", { name: /use file from workspace/i }),
    );
    expect(onUseWorkspaceFile).toHaveBeenCalledTimes(1);
    expect(
      screen.queryByRole("menuitem", { name: /upload from computer/i }),
    ).toBeNull();
  });
});
