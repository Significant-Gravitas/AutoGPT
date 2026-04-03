import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DeleteConfirmationModal } from "../DeleteConfirmationModal";

afterEach(() => {
  cleanup();
});

const credential = { id: "cred-1", title: "My API Key" };

function renderModal(
  overrides: Partial<Parameters<typeof DeleteConfirmationModal>[0]> = {},
) {
  const defaultProps = {
    credentialToDelete: credential,
    isDeleting: false,
    onClose: vi.fn(),
    onConfirm: vi.fn(),
    onForceConfirm: vi.fn(),
    ...overrides,
  };
  return {
    ...render(<DeleteConfirmationModal {...defaultProps} />),
    props: defaultProps,
  };
}

describe("DeleteConfirmationModal", () => {
  it("shows confirmation text with credential title when no warning", () => {
    renderModal();
    expect(screen.getByText(/Are you sure you want to delete/)).toBeDefined();
    expect(screen.getByText(/My API Key/)).toBeDefined();
  });

  it("shows Delete button when no warning message", () => {
    renderModal();
    expect(screen.getByText("Delete")).toBeDefined();
    expect(screen.queryByText("Force Delete")).toBeNull();
  });

  it("shows warning message when provided", () => {
    renderModal({ warningMessage: "Used by 3 agents" });
    expect(screen.getByText("Used by 3 agents")).toBeDefined();
    expect(screen.queryByText(/Are you sure/)).toBeNull();
  });

  it("shows Force Delete button when warning message is present", () => {
    renderModal({ warningMessage: "Credential is in use" });
    expect(screen.getByText("Force Delete")).toBeDefined();
    expect(screen.queryByText("Delete")).toBeNull();
  });

  it("calls onConfirm when Delete button is clicked", () => {
    const { props } = renderModal();
    fireEvent.click(screen.getByText("Delete"));
    expect(props.onConfirm).toHaveBeenCalledOnce();
  });

  it("calls onForceConfirm when Force Delete button is clicked", () => {
    const { props } = renderModal({ warningMessage: "In use" });
    fireEvent.click(screen.getByText("Force Delete"));
    expect(props.onForceConfirm).toHaveBeenCalledOnce();
  });

  it("calls onClose when Cancel button is clicked", () => {
    const { props } = renderModal();
    fireEvent.click(screen.getByText("Cancel"));
    expect(props.onClose).toHaveBeenCalledOnce();
  });

  it("disables Cancel button when isDeleting is true", () => {
    renderModal({ isDeleting: true });
    const cancelButton = screen.getByText("Cancel");
    expect(cancelButton.closest("button")?.disabled).toBe(true);
  });
});
