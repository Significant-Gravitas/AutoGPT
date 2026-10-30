import { after } from "next/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { runAfterResponse } from "../background-tasks";

vi.mock("next/server", () => ({ after: vi.fn() }));

afterEach(() => {
  vi.mocked(after).mockReset();
  vi.restoreAllMocks();
});

describe("runAfterResponse", () => {
  it("hands the task to after without waiting for it", async () => {
    let finish = () => {};
    const task = new Promise<void>((resolve) => {
      finish = resolve;
    });

    runAfterResponse(task);

    expect(after).toHaveBeenCalledOnce();
    const kept = vi.mocked(after).mock.calls[0][0] as Promise<unknown>;
    finish();
    await expect(kept).resolves.toBeUndefined();
  });

  it("logs a failed task instead of letting it reject", async () => {
    const consoleError = vi
      .spyOn(console, "error")
      .mockImplementation(() => {});

    runAfterResponse(Promise.reject(new Error("mailer down")));

    const kept = vi.mocked(after).mock.calls[0][0] as Promise<unknown>;
    await expect(kept).resolves.toBeUndefined();
    expect(consoleError).toHaveBeenCalledWith("Auth background task failed", {
      error: "mailer down",
    });
  });

  it("lets the task run on its own outside a request scope", async () => {
    vi.mocked(after).mockImplementation(() => {
      throw new Error("`after` was called outside a request scope");
    });
    let ran = false;

    expect(() =>
      runAfterResponse(
        Promise.resolve().then(() => {
          ran = true;
        }),
      ),
    ).not.toThrow();

    await vi.waitFor(() => expect(ran).toBe(true));
  });
});
