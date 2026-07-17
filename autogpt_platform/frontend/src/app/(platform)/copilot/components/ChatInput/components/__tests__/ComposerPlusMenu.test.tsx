import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useCopilotModal } from "../../../../useCopilotModal";
import { ComposerPlusMenu } from "../ComposerPlusMenu";

let mockWorkspaceFilesFlag = false;
vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: { CHAT_WORKSPACE_FILES: "chat-workspace-files" },
  useGetFlag: () => mockWorkspaceFilesFlag,
}));

afterEach(() => {
  vi.clearAllMocks();
  mockWorkspaceFilesFlag = false;
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

  it("adds a flat workspace option after Attach file when its flag is enabled", async () => {
    mockWorkspaceFilesFlag = true;
    const onUseWorkspaceFile = vi.fn();
    render(
      <ComposerPlusMenu
        onFilesSelected={vi.fn()}
        onUseWorkspaceFile={onUseWorkspaceFile}
      />,
    );
    openMenu();

    const items = await screen.findAllByRole("menuitem");
    expect(items.map((item) => item.textContent)).toEqual([
      "Attach file",
      "Use File from Workspace",
      "Integrations",
      "Skills",
      "Scheduled",
    ]);

    fireEvent.click(
      screen.getByRole("menuitem", { name: /use file from workspace/i }),
    );
    expect(onUseWorkspaceFile).toHaveBeenCalledTimes(1);
  });

  it("clears the guided prompt for Attach file and Integrations but not Skills or Scheduled", async () => {
    const onClearGuidedPrompt = vi.fn();
    render(
      <ComposerPlusMenu
        onFilesSelected={vi.fn()}
        onClearGuidedPrompt={onClearGuidedPrompt}
      />,
    );

    openMenu();
    fireEvent.click(
      await screen.findByRole("menuitem", { name: /attach file/i }),
    );
    expect(onClearGuidedPrompt).toHaveBeenCalledTimes(1);

    openMenu();
    fireEvent.click(
      await screen.findByRole("menuitem", { name: /integrations/i }),
    );
    expect(onClearGuidedPrompt).toHaveBeenCalledTimes(2);

    openMenu();
    fireEvent.click(await screen.findByRole("menuitem", { name: /skills/i }));
    expect(onClearGuidedPrompt).toHaveBeenCalledTimes(2);

    openMenu();
    fireEvent.click(
      await screen.findByRole("menuitem", { name: /scheduled/i }),
    );
    expect(onClearGuidedPrompt).toHaveBeenCalledTimes(2);
  });

  it("hides the workspace option when its flag is disabled", async () => {
    render(<ComposerPlusMenu onFilesSelected={vi.fn()} />);
    openMenu();

    await screen.findByRole("menuitem", { name: /attach file/i });
    expect(
      screen.queryByRole("menuitem", { name: /use file from workspace/i }),
    ).toBeNull();
  });
});
