import {
  render,
  screen,
  fireEvent,
  waitFor,
  cleanup,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { BlockCostEstimatesContent } from "../components/BlockCostEstimatesContent";

const toastSpy = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: toastSpy }),
}));

const exportSpy = vi.fn();
vi.mock("@/app/api/__generated__/endpoints/admin/admin", () => ({
  getV2ExportBlockCostEstimates: (
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

const sampleResponse = {
  status: 200,
  data: {
    estimates: [
      {
        block_id: "blk-1",
        block_name: "SearchTheWebBlock",
        cost_type: "second",
        samples: 200,
        mean_credits: 7,
        p50_credits: 6,
        p95_credits: 12,
      },
    ],
    total_rows: 1,
    window_days: 7,
    max_window_days: 90,
    min_samples: 10,
    generated_at: "2026-05-07T13:00:00.000Z",
  },
};

describe("BlockCostEstimatesContent", () => {
  test("renders date inputs, min-samples and Aggregate button", () => {
    render(<BlockCostEstimatesContent />);
    expect(screen.getByLabelText(/Start date/i)).toBeDefined();
    expect(screen.getByLabelText(/End date/i)).toBeDefined();
    expect(screen.getByLabelText(/Min samples/i)).toBeDefined();
    expect(screen.getByRole("button", { name: /Aggregate/i })).toBeDefined();
  });

  test("Download JSON is disabled until aggregation succeeds", () => {
    render(<BlockCostEstimatesContent />);
    const download = screen.getByRole("button", {
      name: /Download JSON/i,
    }) as HTMLButtonElement;
    expect(download.disabled).toBe(true);
  });

  test("aggregates and renders the table on success", async () => {
    exportSpy.mockResolvedValue(sampleResponse);
    render(<BlockCostEstimatesContent />);
    fireEvent.click(screen.getByRole("button", { name: /Aggregate/i }));
    await waitFor(() => expect(exportSpy).toHaveBeenCalledTimes(1));
    await waitFor(() =>
      expect(screen.getByText(/SearchTheWebBlock/)).toBeDefined(),
    );
    expect(screen.getByText(/blk-1/)).toBeDefined();
    expect(toastSpy).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Aggregation complete" }),
    );
  });

  test("downloads JSON after a successful aggregation", async () => {
    exportSpy.mockResolvedValue(sampleResponse);
    const createUrl = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:mock");
    const revokeUrl = vi
      .spyOn(URL, "revokeObjectURL")
      .mockImplementation(() => {});
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});

    render(<BlockCostEstimatesContent />);
    fireEvent.click(screen.getByRole("button", { name: /Aggregate/i }));
    await waitFor(() => expect(exportSpy).toHaveBeenCalledTimes(1));
    await waitFor(() => screen.getByText(/SearchTheWebBlock/));

    fireEvent.click(screen.getByRole("button", { name: /Download JSON/i }));
    await waitFor(() => expect(click).toHaveBeenCalledTimes(1));
    expect(createUrl).toHaveBeenCalledTimes(1);

    createUrl.mockRestore();
    revokeUrl.mockRestore();
    click.mockRestore();
  });

  test("surfaces a 400 detail in the toast when the API rejects the window", async () => {
    exportSpy.mockRejectedValue(
      new ApiError("window 200d exceeds max 90d", 400, {
        detail: "window 200d exceeds max 90d",
      }),
    );
    render(<BlockCostEstimatesContent />);
    fireEvent.click(screen.getByRole("button", { name: /Aggregate/i }));
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

  test("falls back to a generic toast on non-ApiError failures", async () => {
    exportSpy.mockRejectedValue(new Error("network blip"));
    render(<BlockCostEstimatesContent />);
    fireEvent.click(screen.getByRole("button", { name: /Aggregate/i }));
    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Aggregation failed",
          description: "network blip",
        }),
      ),
    );
  });
});
