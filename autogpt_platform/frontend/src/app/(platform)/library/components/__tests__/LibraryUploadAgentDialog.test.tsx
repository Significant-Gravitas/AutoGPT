import { describe, expect, test, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@/tests/integrations/test-utils";
import LibraryUploadAgentDialog from "../LibraryUploadAgentDialog/LibraryUploadAgentDialog";
import {
  mockAuthenticatedUser,
  resetAuthState,
} from "@/tests/integrations/helpers/mock-supabase-auth";

describe("LibraryUploadAgentDialog", () => {
  afterEach(() => {
    resetAuthState();
  });

  test("renders upload button", () => {
    mockAuthenticatedUser();
    render(<LibraryUploadAgentDialog />);

    expect(
      screen.getByRole("button", { name: /upload agent/i }),
    ).toBeInTheDocument();
  });

  test("opens dialog when upload button is clicked", async () => {
    mockAuthenticatedUser();
    render(<LibraryUploadAgentDialog />);

    const uploadButton = screen.getByRole("button", { name: /upload agent/i });
    fireEvent.click(uploadButton);

    await waitFor(() => {
      expect(screen.getByText("Upload Agent")).toBeInTheDocument();
    });
  });

  test("dialog contains agent name input", async () => {
    mockAuthenticatedUser();
    render(<LibraryUploadAgentDialog />);

    const uploadButton = screen.getByRole("button", { name: /upload agent/i });
    fireEvent.click(uploadButton);

    await waitFor(() => {
      expect(screen.getByLabelText(/agent name/i)).toBeInTheDocument();
    });
  });

  test("dialog contains agent description input", async () => {
    mockAuthenticatedUser();
    render(<LibraryUploadAgentDialog />);

    const uploadButton = screen.getByRole("button", { name: /upload agent/i });
    fireEvent.click(uploadButton);

    await waitFor(() => {
      expect(screen.getByLabelText(/agent description/i)).toBeInTheDocument();
    });
  });

  test("upload button is disabled when form is incomplete", async () => {
    mockAuthenticatedUser();
    render(<LibraryUploadAgentDialog />);

    const triggerButton = screen.getByRole("button", { name: /upload agent/i });
    fireEvent.click(triggerButton);

    await waitFor(() => {
      const submitButton = screen.getByRole("button", { name: /^upload$/i });
      expect(submitButton).toBeDisabled();
    });
  });

  test("has correct test id on trigger button", () => {
    mockAuthenticatedUser();
    render(<LibraryUploadAgentDialog />);

    expect(screen.getByTestId("upload-agent-button")).toBeInTheDocument();
  });
});
