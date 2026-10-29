import {
  render,
  screen,
  fireEvent,
  waitFor,
  cleanup,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { ExportCreditTransactionsButton } from "../components/ExportCreditTransactionsButton";

const toastSpy = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: toastSpy }),
}));

const exportSpy = vi.fn();
vi.mock("@/app/api/__generated__/endpoints/admin/admin", () => ({
  getV2ExportCreditTransactions: (
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

describe("ExportCreditTransactionsButton", () => {
  test("renders the trigger button", () => {
    render(<ExportCreditTransactionsButton />);
    expect(screen.getByRole("button", { name: /Export CSV/i })).toBeDefined();
  });

  test("opens a dialog with date inputs and a type filter when clicked", async () => {
    render(<ExportCreditTransactionsButton />);
    fireEvent.click(screen.getByRole("button", { name: /Export CSV/i }));
    await waitFor(() => {
      expect(screen.getByLabelText(/Start date/i)).toBeDefined();
      expect(screen.getByLabelText(/End date/i)).toBeDefined();
      expect(screen.getByLabelText(/User ID/i)).toBeDefined();
    });
  });

  test("surfaces a 400 detail in the toast when the API rejects the window", async () => {
    exportSpy.mockRejectedValue(
      new ApiError("Export window must be <= 90 days (got 200.00 days)", 400, {
        detail: "Export window must be <= 90 days (got 200.00 days)",
      }),
    );
    render(<ExportCreditTransactionsButton />);
    fireEvent.click(screen.getByRole("button", { name: /Export CSV/i }));
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

  test("triggers a CSV download on success", async () => {
    exportSpy.mockResolvedValue({
      status: 200,
      data: {
        transactions: [
          {
            transaction_key: "tx-1",
            user_id: "u1",
            user_email: "u1@example.com",
            transaction_time: "2026-04-01T00:00:00Z",
            transaction_type: "GRANT",
            amount: 1000,
            running_balance: 5000,
            current_balance: 5000,
            reason: "Initial grant",
            admin_email: "admin@example.com",
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

    render(<ExportCreditTransactionsButton />);
    fireEvent.click(screen.getByRole("button", { name: /Export CSV/i }));
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

  test("falls back to a generic toast when the API throws a non-ApiError", async () => {
    exportSpy.mockRejectedValue(new Error("network blip"));
    render(<ExportCreditTransactionsButton />);
    fireEvent.click(screen.getByRole("button", { name: /Export CSV/i }));
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
    // Simulate an unexpected non-200 that didn't throw — okData() returns
    // undefined and the operator should see a clear error instead of a crash.
    exportSpy.mockResolvedValue({ status: 204, data: null });
    render(<ExportCreditTransactionsButton />);
    fireEvent.click(screen.getByRole("button", { name: /Export CSV/i }));
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
