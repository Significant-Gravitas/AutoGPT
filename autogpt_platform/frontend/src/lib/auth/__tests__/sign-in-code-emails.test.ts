// @vitest-environment node
import { afterEach, describe, expect, it, vi } from "vitest";
import type { AuthEmailContext } from "../auth-email-cooldown";
import { AuthEmailNotSent } from "../auth-email-not-sent";
import { undeliverableEmailError } from "../email-deliverability";

const sendAuthEmail = vi.hoisted(() => vi.fn());
vi.mock("../email", () => ({ sendAuthEmail }));
const releaseWindowSlot = vi.hoisted(() => vi.fn(async () => {}));
vi.mock("../auth-email-cooldown", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../auth-email-cooldown")>()),
  releaseWindowSlot,
}));

const { sendCode } = await import("../sign-in-code-emails");

const context = {} as AuthEmailContext;
const pending = {
  providerId: "github",
  accountId: "1",
  email: "pat@company.example",
  name: "Pat",
  image: null,
} as const;

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
  sendAuthEmail.mockReset();
  releaseWindowSlot.mockClear();
});

describe("sending a sign-in code", () => {
  it("gives up on a send stuck before its request, and keeps its slot", async () => {
    vi.useFakeTimers();
    vi.spyOn(console, "error").mockImplementation(() => {});
    // Stuck minting the service token, say: it never reaches the request
    // the abort would cancel.
    sendAuthEmail.mockImplementation(() => new Promise(() => {}));

    const sending = sendCode(context, pending, "042917", "slot");
    await vi.advanceTimersByTimeAsync(10_000);

    expect(await sending).toBe(false);
    expect(releaseWindowSlot).not.toHaveBeenCalled();
  });

  it("sends it with a way to cancel the request", async () => {
    sendAuthEmail.mockResolvedValue(undefined);

    expect(await sendCode(context, pending, "042917", "slot")).toBe(true);
    expect(sendAuthEmail).toHaveBeenCalledWith(
      {
        to: "pat@company.example",
        type: "sign_in_code",
        code: "042917",
        provider: "github",
      },
      { signal: expect.any(AbortSignal) },
    );
    expect(releaseWindowSlot).not.toHaveBeenCalled();
  });

  it.each([
    ["the backend refuses the send", new AuthEmailNotSent("mailer down")],
    ["the address takes no mail", undeliverableEmailError()],
  ])("gives the hour's slot back when %s", async (_case, error) => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    sendAuthEmail.mockRejectedValue(error);

    expect(await sendCode(context, pending, "042917", "slot")).toBe(false);
    expect(releaseWindowSlot).toHaveBeenCalledWith(context, "slot");
  });

  it.each([
    // The connection dropped after the request left.
    ["the connection drops", new TypeError("fetch failed")],
    // The backend failed after the mail provider may have taken the email.
    [
      "the backend answers 5xx",
      new Error("Backend auth-email send failed (503)"),
    ],
  ])(
    "keeps the slot when %s: the email may still go out",
    async (_case, error) => {
      vi.spyOn(console, "error").mockImplementation(() => {});
      sendAuthEmail.mockRejectedValue(error);

      expect(await sendCode(context, pending, "042917", "slot")).toBe(false);
      expect(releaseWindowSlot).not.toHaveBeenCalled();
    },
  );

  it("cancels a send the backend hasn't answered in 10 seconds, and keeps its slot", async () => {
    vi.useFakeTimers();
    vi.spyOn(console, "error").mockImplementation(() => {});
    let signal: AbortSignal | undefined;
    sendAuthEmail.mockImplementation(
      (_args: unknown, options: { signal: AbortSignal }) =>
        new Promise((_resolve, reject) => {
          signal = options.signal;
          signal.addEventListener("abort", () =>
            reject(new DOMException("The request was cancelled", "AbortError")),
          );
        }),
    );

    const sending = sendCode(context, pending, "042917", "slot");
    await vi.advanceTimersByTimeAsync(9_999);
    expect(signal?.aborted).toBe(false);
    await vi.advanceTimersByTimeAsync(1);

    expect(await sending).toBe(false);
    expect(signal?.aborted).toBe(true);
    // Its email may still go out, so it still counts toward the hour's five.
    expect(releaseWindowSlot).not.toHaveBeenCalled();
  });
});
