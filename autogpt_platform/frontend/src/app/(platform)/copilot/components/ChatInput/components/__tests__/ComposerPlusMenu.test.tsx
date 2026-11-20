import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useCopilotModal } from "../../../../useCopilotModal";
import { MAX_ATTACHMENTS } from "../../../../helpers/workspaceAttachments";
import { ComposerPlusMenu } from "../ComposerPlusMenu";
import { VoiceIcon } from "@hugeicons/core-free-icons";

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
  it.each([true, false])(
    "respects the disabled state of an extra item: %s",
    async (disabled) => {
      const onStart = vi.fn();
      render(
        <ComposerPlusMenu
          onFilesSelected={vi.fn()}
          extraItems={[
            {
              key: "voice",
              label: "Voice mode",
              icon: VoiceIcon,
              onSelect: onStart,
              disabled,
            },
          ]}
        />,
      );
      openMenu();

      const item = await screen.findByRole("menuitem", { name: "Voice mode" });
      expect(item.getAttribute("aria-disabled")).toBe(disabled ? "true" : null);
      fireEvent.click(item);
      if (disabled) {
        fireEvent.keyDown(item, { key: "Enter" });
        fireEvent.keyDown(item, { key: " " });
      }
      expect(onStart).toHaveBeenCalledTimes(disabled ? 0 : 1);
    },
  );

  it("lists the four actions in fixed order", async () => {
    render(<ComposerPlusMenu onFilesSelected={vi.fn()} />);
    openMenu();

    const items = await screen.findAllByRole("menuitem");
    expect(items.map((item) => item.textContent)).toEqual([
      "Attach file",
      "Connect service",
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

  it("selecting Scheduled and Connect service open their modals", async () => {
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
      await screen.findByRole("menuitem", { name: /connect service/i }),
    );
    expect(screen.getByTestId("modal-probe").textContent).toBe("connect");
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
      "Connect service",
      "Skills",
      "Scheduled",
    ]);

    fireEvent.click(
      screen.getByRole("menuitem", { name: /use file from workspace/i }),
    );
    expect(onUseWorkspaceFile).toHaveBeenCalledTimes(1);
  });

  it("clears the guided prompt for Attach file and Connect service but not Skills or Scheduled", async () => {
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
      await screen.findByRole("menuitem", { name: /connect service/i }),
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

  it("disables both file entries at the cap and says what the cap is", async () => {
    mockWorkspaceFilesFlag = true;
    const onFilesSelected = vi.fn();
    const onUseWorkspaceFile = vi.fn();
    render(
      <ComposerPlusMenu
        onFilesSelected={onFilesSelected}
        onUseWorkspaceFile={onUseWorkspaceFile}
        isAtCap
      />,
    );
    openMenu();

    const items = await screen.findAllByRole("menuitem");
    const disabled = items.filter(
      (item) => item.getAttribute("data-disabled") !== null,
    );
    expect(disabled.map((item) => item.textContent)).toEqual([
      `Attach file${MAX_ATTACHMENTS} max`,
      `Use File from Workspace${MAX_ATTACHMENTS} max`,
    ]);

    fireEvent.click(
      screen.getByRole("menuitem", { name: /use file from workspace/i }),
    );
    expect(onUseWorkspaceFile).not.toHaveBeenCalled();
  });

  it("leaves the non-file entries alone at the cap", async () => {
    render(<ComposerPlusMenu onFilesSelected={vi.fn()} isAtCap />);
    openMenu();

    fireEvent.click(await screen.findByRole("menuitem", { name: /skills/i }));

    expect(screen.queryByRole("menuitem")).toBeNull();
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

describe("ComposerPlusMenu camera capture", () => {
  function stubCoarsePointer(matches: boolean) {
    vi.stubGlobal(
      "matchMedia",
      vi.fn().mockImplementation((query: string) => ({
        matches: query === "(pointer: coarse)" && matches,
        media: query,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
      })),
    );
  }

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("offers Take photo right after Attach file on touch devices", async () => {
    stubCoarsePointer(true);
    render(<ComposerPlusMenu onFilesSelected={vi.fn()} />);
    openMenu();

    const items = await screen.findAllByRole("menuitem");
    expect(items.slice(0, 2).map((item) => item.textContent)).toEqual([
      "Attach file",
      "Take photo",
    ]);
  });

  it("hides Take photo when the primary pointer is not touch", async () => {
    stubCoarsePointer(false);
    render(<ComposerPlusMenu onFilesSelected={vi.fn()} />);
    openMenu();

    await screen.findAllByRole("menuitem");
    expect(screen.queryByRole("menuitem", { name: /take photo/i })).toBeNull();
  });

  it("uses a rear-camera image input and hands captured photos over", async () => {
    stubCoarsePointer(true);
    const onFilesSelected = vi.fn();
    render(<ComposerPlusMenu onFilesSelected={onFilesSelected} />);

    const input = screen.getByTestId("composer-camera-input");
    expect(input.getAttribute("capture")).toBe("environment");
    expect(input.getAttribute("accept")).toBe("image/*");

    const photo = new File(["x"], "image.jpg", { type: "image/jpeg" });
    fireEvent.change(input, { target: { files: [photo] } });

    await waitFor(() => expect(onFilesSelected).toHaveBeenCalledTimes(1));
    const [files] = onFilesSelected.mock.calls[0];
    expect(files).toHaveLength(1);
    expect(files[0].name).toMatch(/^photo-.*\.jpg$/);
  });

  it("disables Take photo at the attachment cap", async () => {
    stubCoarsePointer(true);
    render(<ComposerPlusMenu onFilesSelected={vi.fn()} isAtCap />);
    openMenu();

    const item = await screen.findByRole("menuitem", { name: /take photo/i });
    expect(item.getAttribute("data-disabled")).not.toBeNull();
  });
});
