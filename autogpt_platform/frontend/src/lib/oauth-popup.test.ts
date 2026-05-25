import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import {
  OAUTH_ERROR_FLOW_CANCELED,
  OAUTH_ERROR_WINDOW_CLOSED,
  openOAuthPopup,
} from "./oauth-popup";

// Minimal popup stub — window.open returns this. `closed` flips when the
// "user" closes the popup so the close-detect interval can observe it.
function makePopupStub() {
  return {
    closed: false,
    location: { href: "" },
    close: vi.fn(),
  };
}

function setupPopup(stub: ReturnType<typeof makePopupStub> | null) {
  return vi
    .spyOn(window, "open")
    .mockImplementation(() => stub as unknown as Window);
}

describe("openOAuthPopup popup-close grace window", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    localStorage.clear();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  test("rejects with WINDOW_CLOSED after grace if no result arrives", async () => {
    const popup = makePopupStub();
    setupPopup(popup);

    const { promise } = openOAuthPopup("https://example.com/oauth", {
      stateToken: "tok-1",
      useCrossOriginListeners: true,
    });
    const onReject = vi.fn();
    promise.catch(onReject);

    // User closes the popup.
    popup.closed = true;

    // First closed-poll tick (500ms) observes closed and starts the 3s grace.
    await vi.advanceTimersByTimeAsync(500);
    expect(onReject).not.toHaveBeenCalled();

    // Mid-grace: still pending.
    await vi.advanceTimersByTimeAsync(1500);
    expect(onReject).not.toHaveBeenCalled();

    // Grace expires (total +3000ms after close-detect) → reject fires.
    await vi.advanceTimersByTimeAsync(1600);
    expect(onReject).toHaveBeenCalledTimes(1);
    expect(onReject.mock.calls[0][0]).toBeInstanceOf(Error);
    expect((onReject.mock.calls[0][0] as Error).message).toBe(
      OAUTH_ERROR_WINDOW_CLOSED,
    );
  });

  test("final localStorage sweep resolves when result lands after close", async () => {
    const popup = makePopupStub();
    setupPopup(popup);

    const { promise } = openOAuthPopup("https://example.com/oauth", {
      stateToken: "tok-2",
      useCrossOriginListeners: true,
    });
    const onResolve = vi.fn();
    const onReject = vi.fn();
    promise.then(onResolve, onReject);

    // Result lands in scoped localStorage just before the user closes the
    // popup — the BroadcastChannel listener never fired (storage partitioning)
    // and the periodic poll hasn't ticked yet.
    localStorage.setItem(
      "oauth_popup_result_tok-2",
      JSON.stringify({
        message_type: "mcp_oauth_result",
        success: true,
        code: "auth-code-xyz",
        state: "tok-2",
      }),
    );
    popup.closed = true;

    // First closed-poll tick runs the synchronous final-storage check,
    // which resolves the promise before the grace timer even arms.
    await vi.advanceTimersByTimeAsync(500);

    expect(onReject).not.toHaveBeenCalled();
    expect(onResolve).toHaveBeenCalledWith({
      code: "auth-code-xyz",
      state: "tok-2",
    });
    // Storage entry consumed.
    expect(localStorage.getItem("oauth_popup_result_tok-2")).toBeNull();
  });

  test("result arriving during grace window cancels the WINDOW_CLOSED reject", async () => {
    const popup = makePopupStub();
    setupPopup(popup);

    const { promise } = openOAuthPopup("https://example.com/oauth", {
      stateToken: "tok-3",
      useCrossOriginListeners: true,
    });
    const onResolve = vi.fn();
    const onReject = vi.fn();
    promise.then(onResolve, onReject);

    // User closes popup before the result lands.
    popup.closed = true;
    await vi.advanceTimersByTimeAsync(500); // close-detect fires, grace armed

    // Result lands ~1s into the grace via localStorage (polled every 500ms).
    localStorage.setItem(
      "oauth_popup_result_tok-3",
      JSON.stringify({
        message_type: "mcp_oauth_result",
        success: true,
        code: "late-code",
        state: "tok-3",
      }),
    );
    await vi.advanceTimersByTimeAsync(1000);

    expect(onResolve).toHaveBeenCalledWith({
      code: "late-code",
      state: "tok-3",
    });

    // Run out the rest of the grace window — must NOT reject after the fact.
    await vi.advanceTimersByTimeAsync(3000);
    expect(onReject).not.toHaveBeenCalled();
  });

  test("abort during grace window tears down the grace timer", async () => {
    const popup = makePopupStub();
    setupPopup(popup);

    const { promise, cleanup } = openOAuthPopup("https://example.com/oauth", {
      stateToken: "tok-4",
      useCrossOriginListeners: true,
    });
    const onReject = vi.fn();
    promise.catch(onReject);

    popup.closed = true;
    await vi.advanceTimersByTimeAsync(500); // grace armed

    // Caller aborts (e.g. component unmount) before grace expires.
    cleanup.abort();
    await vi.advanceTimersByTimeAsync(10);

    // Abort wins → CANCELED, not WINDOW_CLOSED.
    expect(onReject).toHaveBeenCalledTimes(1);
    expect((onReject.mock.calls[0][0] as Error).message).toBe(
      OAUTH_ERROR_FLOW_CANCELED,
    );

    // Advancing past the original grace deadline must not produce a second
    // reject — the grace setTimeout was cleared by the abort listener.
    await vi.advanceTimersByTimeAsync(5000);
    expect(onReject).toHaveBeenCalledTimes(1);
  });

  test("same-origin flow rejects immediately on popup close (no grace)", async () => {
    const popup = makePopupStub();
    setupPopup(popup);

    const { promise } = openOAuthPopup("https://example.com/oauth", {
      stateToken: "tok-so",
      useCrossOriginListeners: false, // same-origin → no grace
    });
    const onReject = vi.fn();
    promise.catch(onReject);

    popup.closed = true;
    // One close-poll tick (500ms) is enough — there's no 3s grace
    // because synchronous opener.postMessage would have already fired.
    await vi.advanceTimersByTimeAsync(500);

    expect(onReject).toHaveBeenCalledTimes(1);
    expect((onReject.mock.calls[0][0] as Error).message).toBe(
      OAUTH_ERROR_WINDOW_CLOSED,
    );
  });

  test("overall timeout rejects with FLOW_TIMED_OUT if popup never closes", async () => {
    const popup = makePopupStub();
    setupPopup(popup);

    const { promise } = openOAuthPopup("https://example.com/oauth", {
      stateToken: "tok-timeout",
      useCrossOriginListeners: true,
      timeout: 1000, // tight outer timeout for the test
    });
    const onReject = vi.fn();
    promise.catch(onReject);

    // Popup never closes → close-poll never observes ``closed`` → only
    // the outer timeout can reject.
    await vi.advanceTimersByTimeAsync(1100);

    expect(onReject).toHaveBeenCalledTimes(1);
    expect((onReject.mock.calls[0][0] as Error).message).toMatch(/timed out/i);
  });

  test("state-mismatch message is ignored and does not resolve the promise", async () => {
    const popup = makePopupStub();
    setupPopup(popup);

    const { promise, cleanup } = openOAuthPopup("https://example.com/oauth", {
      stateToken: "expected-state",
      useCrossOriginListeners: false,
    });
    const onResolve = vi.fn();
    const onReject = vi.fn();
    promise.then(onResolve, onReject);

    // Dispatch a postMessage with a different state token — must be ignored.
    window.dispatchEvent(
      new MessageEvent("message", {
        data: {
          message_type: "oauth_popup_result",
          success: true,
          code: "wrong-code",
          state: "OTHER-state",
        },
      }),
    );

    await vi.advanceTimersByTimeAsync(10);
    expect(onResolve).not.toHaveBeenCalled();
    expect(onReject).not.toHaveBeenCalled();

    cleanup.abort();
  });
});
