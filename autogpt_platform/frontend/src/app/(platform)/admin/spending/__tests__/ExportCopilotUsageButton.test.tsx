import {
  render,
  screen,
  fireEvent,
  waitFor,
  cleanup,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { ExportCopilotUsageButton } from "../components/ExportCopilotUsageButton";

const toastSpy = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: toastSpy }),
}));

const exportSpy = vi.fn();
vi.mock("@/app/api/__generated__/endpoints/admin/admin", () => ({
  getV2ExportCopilotWeeklyUsageVsRateLimit: (
    ...args: Parameters<typeof exportSpy>
  ): ReturnType<typeof exportSpy> => exportSpy(...args),
}));

beforeEach(() => {
  toastSpy.mockReset();
  exportSpy.mockReset();
});

afterEach(() => {
  cleanup();
});

describe("ExportCopilotUsageButton", () => {
  test("renders the trigger button", () => {
    render(<ExportCopilotUsageButton />);
    expect(
      screen.getByRole("button", { name: /Copilot Usage CSV/i }),
    ).toBeDefined();
  });

  test("opens a dialog with date inputs when clicked", async () => {
    render(<ExportCopilotUsageButton />);
    fireEvent.click(screen.getByRole("button", { name: /Copilot Usage CSV/i }));
    await waitFor(() => {
      expect(screen.getByLabelText(/Start date/i)).toBeDefined();
      expect(screen.getByLabelText(/End date/i)).toBeDefined();
    });
  });

  test("triggers a CSV download on success", async () => {
    exportSpy.mockResolvedValue({
      status: 200,
      data: {
        rows: [
          {
            user_id: "u1",
            user_email: "u1@example.com",
            week_start: "2026-03-30T00:00:00Z",
            week_end: "2026-04-05T23:59:59.999Z",
            copilot_cost_microdollars: 1_500_000,
            tier: "PRO",
            weekly_limit_microdollars: 25_000_000,
            percent_used: 6.0,
          },
        ],
        total_rows: 1,
        window_days: 30,
        max_window_days: 90,
      },
    });
    const createUrl = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:mock");
    const revokeUrl = vi
      .spyOn(URL, "revokeObjectURL")
      .mockImplementation(() => {});
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});

    render(<ExportCopilotUsageButton />);
    fireEvent.click(screen.getByRole("button", { name: /Copilot Usage CSV/i }));
    await waitFor(() => screen.getByLabelText(/Start date/i));
    fireEvent.click(screen.getByRole("button", { name: /Download CSV/i }));

    await waitFor(() => expect(click).toHaveBeenCalledTimes(1));
    expect(createUrl).toHaveBeenCalledTimes(1);
    expect(toastSpy).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Export ready" }),
    );

    createUrl.mockRestore();
    revokeUrl.mockRestore();
    click.mockRestore();
  });

  test("surfaces 400 detail as a toast when the window is too large", async () => {
    exportSpy.mockRejectedValue(
      new ApiError("Export window must be <= 90 days (got 200.00 days)", 400, {
        detail: "Export window must be <= 90 days (got 200.00 days)",
      }),
    );
    render(<ExportCopilotUsageButton />);
    fireEvent.click(screen.getByRole("button", { name: /Copilot Usage CSV/i }));
    await waitFor(() => screen.getByLabelText(/Start date/i));
    fireEvent.click(screen.getByRole("button", { name: /Download CSV/i }));
    await waitFor(() => expect(exportSpy).toHaveBeenCalledTimes(1));
    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Window too large",
          variant: "destructive",
        }),
      ),
    );
  });

  test("falls back to a generic toast when the API throws a non-ApiError", async () => {
    exportSpy.mockRejectedValue(new Error("network blip"));
    render(<ExportCopilotUsageButton />);
    fireEvent.click(screen.getByRole("button", { name: /Copilot Usage CSV/i }));
    await waitFor(() => screen.getByLabelText(/Start date/i));
    fireEvent.click(screen.getByRole("button", { name: /Download CSV/i }));
    await waitFor(() => expect(exportSpy).toHaveBeenCalledTimes(1));
    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Export failed",
          description: "network blip",
          variant: "destructive",
        }),
      ),
    );
  });

  test("warns when the response is missing the expected success shape", async () => {
    exportSpy.mockResolvedValue({ status: 204, data: null });
    render(<ExportCopilotUsageButton />);
    fireEvent.click(screen.getByRole("button", { name: /Copilot Usage CSV/i }));
    await waitFor(() => screen.getByLabelText(/Start date/i));
    fireEvent.click(screen.getByRole("button", { name: /Download CSV/i }));
    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Export failed",
          variant: "destructive",
        }),
      ),
    );
  });
});
