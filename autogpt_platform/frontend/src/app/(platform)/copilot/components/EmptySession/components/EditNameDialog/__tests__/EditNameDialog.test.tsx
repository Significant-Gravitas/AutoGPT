import { beforeEach, describe, expect, test, vi } from "vitest";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { http, HttpResponse } from "msw";
import { EditNameDialog } from "../EditNameDialog";

const mockToast = vi.hoisted(() => vi.fn());
const mockRefreshSession = vi.hoisted(() => vi.fn());

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: mockToast }),
}));

vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: () => ({
    refreshSession: mockRefreshSession,
  }),
}));

function mockUpdateNameSuccess() {
  server.use(
    http.put("/api/auth/user", () => {
      return HttpResponse.json({ user: { id: "u1" } });
    }),
  );
}

function mockUpdateNameError(message = "Network error") {
  server.use(
    http.put("/api/auth/user", () => {
      return HttpResponse.json({ error: message }, { status: 400 });
    }),
  );
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
    mockRefreshSession.mockReset();
    mockRefreshSession.mockResolvedValue({ user: { id: "u1" } });
  });

  test("opens dialog with current name prefilled", async () => {
    mockUpdateNameSuccess();
    render(<EditNameDialog currentName="Alice" />);

    const input = await openDialogAndGetInput();
    expect(input.value).toBe("Alice");
  });

  test("saves name via API route and closes dialog", async () => {
    mockUpdateNameSuccess();
    render(<EditNameDialog currentName="Alice" />);

    const input = await openDialogAndGetInput();
    fireEvent.change(input, { target: { value: "Bob" } });
    fireEvent.click(getSaveButton());

    await waitFor(() => {
      expect(mockRefreshSession).toHaveBeenCalled();
    });
    expect(mockToast).toHaveBeenCalledWith({ title: "Name updated" });
  });

  test("shows error toast when API returns error", async () => {
    mockUpdateNameError("Network error");
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
    expect(mockRefreshSession).not.toHaveBeenCalled();
  });

  test("shows warning toast when refreshSession returns an error", async () => {
    mockUpdateNameSuccess();
    mockRefreshSession.mockResolvedValue({ error: "refresh failed" });

    render(<EditNameDialog currentName="Alice" />);

    const input = await openDialogAndGetInput();
    fireEvent.change(input, { target: { value: "Bob" } });
    fireEvent.click(getSaveButton());

    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Name saved, but session refresh failed",
          description: "refresh failed",
          variant: "destructive",
        }),
      );
    });
    expect(mockToast).not.toHaveBeenCalledWith({ title: "Name updated" });
  });

  test("disables Save button while input is empty", async () => {
    mockUpdateNameSuccess();
    render(<EditNameDialog currentName="Alice" />);

    const input = await openDialogAndGetInput();
    fireEvent.change(input, { target: { value: "   " } });

    expect(getSaveButton().disabled).toBe(true);
  });
});
