import { beforeEach, describe, expect, test, vi } from "vitest";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { EditNameDialog } from "../EditNameDialog";

const mockToast = vi.hoisted(() => vi.fn());
const mockUseSupabase = vi.hoisted(() => vi.fn());

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: mockToast }),
}));

vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: mockUseSupabase,
}));

function setup({
  updateUser = vi.fn().mockResolvedValue({ error: null }),
  refreshSession = vi.fn().mockResolvedValue(undefined),
}: {
  updateUser?: ReturnType<typeof vi.fn>;
  refreshSession?: ReturnType<typeof vi.fn>;
} = {}) {
  mockUseSupabase.mockReturnValue({
    supabase: { auth: { updateUser } },
    refreshSession,
  });
  return { updateUser, refreshSession };
}

async function openDialogAndGetInput() {
  const trigger = screen.getByRole("button");
  fireEvent.click(trigger);
  await screen.findAllByLabelText(/display name/i);
  const inputs =
    document.querySelectorAll<HTMLInputElement>("input#display-name");
  return inputs[0];
}

function getSaveButton() {
  const saves = screen.getAllByRole("button", { name: /save/i });
  return saves[0] as HTMLButtonElement;
}

describe("EditNameDialog", () => {
  beforeEach(() => {
    mockToast.mockReset();
    mockUseSupabase.mockReset();
  });

  test("opens dialog with current name prefilled", async () => {
    setup();
    render(<EditNameDialog currentName="Alice" />);

    const input = await openDialogAndGetInput();
    expect(input.value).toBe("Alice");
  });

  test("saves name successfully and closes dialog", async () => {
    const { updateUser, refreshSession } = setup();
    render(<EditNameDialog currentName="Alice" />);

    const input = await openDialogAndGetInput();
    fireEvent.change(input, { target: { value: "Bob" } });
    fireEvent.click(getSaveButton());

    await waitFor(() => {
      expect(updateUser).toHaveBeenCalledWith({ data: { full_name: "Bob" } });
    });
    expect(refreshSession).toHaveBeenCalled();
    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith({ title: "Name updated" });
    });
  });

  test("shows error toast when updateUser fails and keeps dialog open", async () => {
    const updateUser = vi
      .fn()
      .mockResolvedValue({ error: { message: "Network error" } });
    const refreshSession = vi.fn();
    setup({ updateUser, refreshSession });

    render(<EditNameDialog currentName="Alice" />);

    const input = await openDialogAndGetInput();
    fireEvent.change(input, { target: { value: "Bob" } });
    fireEvent.click(getSaveButton());

    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Failed to update name",
          description: "Network error",
          variant: "destructive",
        }),
      );
    });
    expect(refreshSession).not.toHaveBeenCalled();
  });

  test("closes dialog and toasts failure when refreshSession throws", async () => {
    const updateUser = vi.fn().mockResolvedValue({ error: null });
    const refreshSession = vi
      .fn()
      .mockRejectedValue(new Error("refresh failed"));
    setup({ updateUser, refreshSession });

    render(<EditNameDialog currentName="Alice" />);

    const input = await openDialogAndGetInput();
    fireEvent.change(input, { target: { value: "Bob" } });
    fireEvent.click(getSaveButton());

    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Name saved, but session refresh failed",
          variant: "destructive",
        }),
      );
    });
    expect(mockToast).not.toHaveBeenCalledWith({ title: "Name updated" });
  });

  test("disables Save button while empty input", async () => {
    setup();
    render(<EditNameDialog currentName="Alice" />);

    const input = await openDialogAndGetInput();
    fireEvent.change(input, { target: { value: "   " } });

    expect(getSaveButton().disabled).toBe(true);
  });
});
