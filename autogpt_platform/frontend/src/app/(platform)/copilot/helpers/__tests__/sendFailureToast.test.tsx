import { toast } from "@/components/molecules/Toast/use-toast";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SentNowError } from "../../components/SendAtReset/helpers";
import { ReportedSendError } from "../reportedSendError";
import { toastSendFailure } from "../sendFailureToast";

vi.mock("@/components/molecules/Toast/use-toast", () => ({ toast: vi.fn() }));

const ARGS = {
  title: "Couldn't send message",
  recovery: "your message is back in the composer",
};

// Send at reset sends at once when the budget turns out to be back, and
// reports that send's failure under its own title.
describe("a send Send at reset made at once", () => {
  beforeEach(() => {
    vi.mocked(toast).mockClear();
  });

  it("adds no toast to a refusal the chat already showed", () => {
    const shown = new ReportedSendError("The plan dialog is open");

    toastSendFailure({ ...ARGS, error: new SentNowError(shown) });

    expect(toast).not.toHaveBeenCalled();
  });

  it("offers plans when the account needs one", () => {
    const paywall = new ApiError("Payment required", 402, {});

    toastSendFailure({ ...ARGS, error: new SentNowError(paywall) });

    expect(vi.mocked(toast).mock.calls[0][0]).toMatchObject({
      title: "Sending needs an active plan",
    });
  });
});
