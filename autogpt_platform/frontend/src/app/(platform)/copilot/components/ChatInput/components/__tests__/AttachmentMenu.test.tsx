import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AttachmentMenu } from "../AttachmentMenu";

afterEach(() => {
  vi.clearAllMocks();
});

describe("AttachmentMenu", () => {
  it("renders a plain attach button when the workspace option is hidden", () => {
    render(
      <AttachmentMenu onFilesSelected={vi.fn()} showWorkspaceOption={false} />,
    );
    expect(screen.getByRole("button", { name: /attach file/i })).toBeTruthy();
    // No dropdown menu items are exposed in the plain variant.
    expect(screen.queryByText(/use file from workspace/i)).toBeNull();
  });

  it("exposes both upload sources when the workspace option is enabled", async () => {
    render(
      <AttachmentMenu onFilesSelected={vi.fn()} showWorkspaceOption={true} />,
    );
    fireEvent.pointerDown(
      screen.getByRole("button", { name: /attach file/i }),
      { button: 0 },
    );
    expect(
      await screen.findByRole("menuitem", { name: /upload from computer/i }),
    ).toBeTruthy();
    expect(
      screen.getByRole("menuitem", { name: /use file from workspace/i }),
    ).toBeTruthy();
  });

  it("invokes onUseWorkspaceFile when the workspace item is chosen", async () => {
    const onUseWorkspaceFile = vi.fn();
    render(
      <AttachmentMenu
        onFilesSelected={vi.fn()}
        onUseWorkspaceFile={onUseWorkspaceFile}
        showWorkspaceOption={true}
      />,
    );
    fireEvent.pointerDown(
      screen.getByRole("button", { name: /attach file/i }),
      { button: 0 },
    );
    fireEvent.click(
      await screen.findByRole("menuitem", { name: /use file from workspace/i }),
    );
    expect(onUseWorkspaceFile).toHaveBeenCalledTimes(1);
  });
});
