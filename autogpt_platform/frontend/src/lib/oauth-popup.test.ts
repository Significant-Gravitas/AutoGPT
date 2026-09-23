import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import {
  OAUTH_ERROR_FLOW_CANCELED,
  OAUTH_ERROR_POPUP_BLOCKED_NO_TAB,
  OAUTH_ERROR_WINDOW_CLOSED,
  openOAuthPopup,
  preOpenOAuthPopup,
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

describe("openOAuthPopup popup-close handling", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    localStorage.clear();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  test("cross-origin flow survives a COOP-severed handle reporting closed", async () => {
    const popup = makePopupStub();
    setupPopup(popup);

    const { promise } = openOAuthPopup("https://example.com/oauth", {
      stateToken: "tok-1",
      useCrossOriginListeners: true,
    });
    const onResolve = vi.fn();
    const onReject = vi.fn();
    promise.then(onResolve, onReject);

    // Providers serving COOP: same-origin (e.g. Stripe) trigger a
    // browsing-context-group swap when the popup navigates to them — the
    // parent's handle is severed and ``closed`` flips to true while the
    // window is still open and the user hasn't signed in yet.
    popup.closed = true;

    // The user takes far longer than any close-based deadline to authorize.
    await vi.advanceTimersByTimeAsync(30_000);
    expect(onReject).not.toHaveBeenCalled();

    // The callback page finally lands the result via localStorage.
    localStorage.setItem(
      "oauth_popup_result_tok-1",
      JSON.stringify({
        message_type: "mcp_oauth_result",
        success: true,
        code: "late-auth-code",
        state: "tok-1",
      }),
    );
    await vi.advanceTimersByTimeAsync(500);

    expect(onReject).not.toHaveBeenCalled();
    expect(onResolve).toHaveBeenCalledWith({
      code: "late-auth-code",
      state: "tok-1",
    });
  });

  test("cross-origin localStorage poll resolves the flow", async () => {
    const popup = makePopupStub();
    setupPopup(popup);

    const { promise } = openOAuthPopup("https://example.com/oauth", {
      stateToken: "tok-2",
      useCrossOriginListeners: true,
    });
    const onResolve = vi.fn();
    const onReject = vi.fn();
    promise.then(onResolve, onReject);

    // The BroadcastChannel listener never fires, so the callback page leaves
    // the result in scoped localStorage for the periodic poll to read.
    localStorage.setItem(
      "oauth_popup_result_tok-2",
      JSON.stringify({
        message_type: "mcp_oauth_result",
        success: true,
        code: "auth-code-xyz",
        state: "tok-2",
      }),
    );

    // The next periodic localStorage poll resolves the flow.
    await vi.advanceTimersByTimeAsync(500);

    expect(onReject).not.toHaveBeenCalled();
    expect(onResolve).toHaveBeenCalledWith({
      code: "auth-code-xyz",
      state: "tok-2",
    });
    // Storage entry consumed.
    expect(localStorage.getItem("oauth_popup_result_tok-2")).toBeNull();
  });

  test("result arriving after the handle reports closed still resolves", async () => {
    const popup = makePopupStub();
    setupPopup(popup);

    const { promise } = openOAuthPopup("https://example.com/oauth", {
      stateToken: "tok-3",
      useCrossOriginListeners: true,
    });
    const onResolve = vi.fn();
    const onReject = vi.fn();
    promise.then(onResolve, onReject);

    // Handle reports closed before the result lands.
    popup.closed = true;
    await vi.advanceTimersByTimeAsync(500);

    // Result lands ~1s later via localStorage (polled every 500ms).
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

    // Keep advancing — must NOT reject after the fact.
    await vi.advanceTimersByTimeAsync(3000);
    expect(onReject).not.toHaveBeenCalled();
  });

  test("abort after the handle reports closed rejects with CANCELED", async () => {
    const popup = makePopupStub();
    setupPopup(popup);

    const { promise, cleanup } = openOAuthPopup("https://example.com/oauth", {
      stateToken: "tok-4",
      useCrossOriginListeners: true,
    });
    const onReject = vi.fn();
    promise.catch(onReject);

    popup.closed = true;
    await vi.advanceTimersByTimeAsync(500);

    // Caller aborts (e.g. component unmount).
    cleanup.abort();
    await vi.advanceTimersByTimeAsync(10);

    // Abort wins → CANCELED, not WINDOW_CLOSED.
    expect(onReject).toHaveBeenCalledTimes(1);
    expect((onReject.mock.calls[0][0] as Error).message).toBe(
      OAUTH_ERROR_FLOW_CANCELED,
    );

    // Advancing further must not produce a second reject.
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

  test("timeout cancels provider-side pending state exactly once", async () => {
    const popup = makePopupStub();
    setupPopup(popup);
    const fetchMock = vi
      .fn()
      .mockResolvedValue(new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);

    const { promise } = openOAuthPopup("https://example.com/oauth", {
      stateToken: "tok-server-timeout",
      cancelUrl: "/api/oauth/pending/cancel",
      timeout: 1000,
    });
    promise.catch(() => {});

    await vi.advanceTimersByTimeAsync(1100);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/oauth/pending/cancel"),
      expect.objectContaining({ method: "POST", keepalive: true }),
    );
  });

  test("manual abort cancels provider-side pending state exactly once", async () => {
    const popup = makePopupStub();
    setupPopup(popup);
    const fetchMock = vi
      .fn()
      .mockResolvedValue(new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);

    const { promise, cleanup } = openOAuthPopup("https://example.com/oauth", {
      stateToken: "tok-server-abort",
      cancelUrl: "/api/oauth/pending/cancel",
    });
    promise.catch(() => {});

    cleanup.abort();
    cleanup.abort();
    await vi.runAllTicks();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(popup.close).toHaveBeenCalledTimes(1);
  });

  test("popup close cancels provider-side pending state exactly once", async () => {
    const popup = makePopupStub();
    setupPopup(popup);
    const fetchMock = vi
      .fn()
      .mockResolvedValue(new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);

    const { promise, cleanup } = openOAuthPopup("https://example.com/oauth", {
      stateToken: "tok-server-close",
      cancelUrl: "/api/oauth/pending/cancel",
    });
    promise.catch(() => {});

    popup.closed = true;
    await vi.advanceTimersByTimeAsync(500);
    cleanup.abort();

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  test("successful result does not cancel provider-side state", async () => {
    const popup = makePopupStub();
    setupPopup(popup);
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    const { promise } = openOAuthPopup("https://example.com/oauth", {
      stateToken: "tok-success",
      cancelUrl: "/api/oauth/pending/cancel",
    });
    window.dispatchEvent(
      new MessageEvent("message", {
        data: {
          message_type: "oauth_popup_result",
          success: true,
          code: "auth-code",
          state: "tok-success",
        },
      }),
    );

    await expect(promise).resolves.toEqual({
      code: "auth-code",
      state: "tok-success",
    });
    expect(fetchMock).not.toHaveBeenCalled();
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

describe("preOpenedWindow option", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    localStorage.clear();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  test("navigates the pre-opened window instead of calling window.open again", () => {
    const openSpy = vi.spyOn(window, "open");
    const preOpened = makePopupStub();

    const { promise, cleanup, popupBlocked, fallbackBlocked } = openOAuthPopup(
      "https://example.com/oauth",
      {
        stateToken: "tok-pre",
        preOpenedWindow: preOpened as unknown as Window,
      },
    );
    promise.catch(() => {});

    expect(openSpy).not.toHaveBeenCalled();
    expect(preOpened.location.href).toBe("https://example.com/oauth");
    expect(popupBlocked).toBe(false);
    expect(fallbackBlocked).toBe(false);

    // After adoption the helper owns the window: aborting must close it —
    // callers no longer close an adopted window themselves.
    cleanup.abort();
    expect(preOpened.close).toHaveBeenCalled();
  });

  test("already-closed preOpenedWindow goes to the new-tab fallback", () => {
    const openSpy = setupPopup(makePopupStub());
    const preOpened = makePopupStub();
    preOpened.closed = true;

    const { promise, cleanup, popupBlocked, fallbackBlocked } = openOAuthPopup(
      "https://example.com/oauth",
      {
        stateToken: "tok-pre-closed",
        preOpenedWindow: preOpened as unknown as Window,
      },
    );
    promise.catch(() => {});

    expect(popupBlocked).toBe(true);
    expect(fallbackBlocked).toBe(false);
    expect(openSpy).toHaveBeenCalledTimes(1);
    expect(openSpy).toHaveBeenCalledWith("https://example.com/oauth", "_blank");

    cleanup.abort();
  });

  test("null preOpenedWindow goes straight to the new-tab fallback", () => {
    const fallback = makePopupStub();
    const openSpy = setupPopup(fallback);

    const { promise, cleanup, popupBlocked, fallbackBlocked } = openOAuthPopup(
      "https://example.com/oauth",
      {
        stateToken: "tok-null",
        preOpenedWindow: null,
      },
    );
    promise.catch(() => {});

    expect(popupBlocked).toBe(true);
    expect(fallbackBlocked).toBe(false);
    // Only the fallback open fires, with the real login URL.
    expect(openSpy).toHaveBeenCalledTimes(1);
    expect(openSpy).toHaveBeenCalledWith("https://example.com/oauth", "_blank");

    cleanup.abort();
    expect(fallback.close).toHaveBeenCalledOnce();
  });

  test("preOpenOAuthPopup opens a blank popup window", () => {
    const popup = makePopupStub();
    const openSpy = setupPopup(popup);

    const result = preOpenOAuthPopup();

    expect(result).toBe(popup);
    expect(openSpy).toHaveBeenCalledTimes(1);
    expect(openSpy.mock.calls[0][0]).toBe("about:blank");
  });

  test("rejects immediately when the new-tab fallback is blocked too", async () => {
    // iOS Safari case: the synchronous pre-open was already blocked, and the
    // fallback open after the async break has no gesture context either.
    setupPopup(null);

    const { promise, popupBlocked, fallbackBlocked } = openOAuthPopup(
      "https://example.com/oauth",
      {
        stateToken: "tok-blocked",
        preOpenedWindow: null,
      },
    );
    const onReject = vi.fn();
    promise.catch(onReject);

    await vi.advanceTimersByTimeAsync(10);

    expect(popupBlocked).toBe(true);
    expect(fallbackBlocked).toBe(true);
    expect(onReject).toHaveBeenCalledTimes(1);
    expect((onReject.mock.calls[0][0] as Error).message).toBe(
      OAUTH_ERROR_POPUP_BLOCKED_NO_TAB,
    );
  });

  test("blocked popup and fallback cancel provider-side state exactly once", async () => {
    setupPopup(null);
    const fetchMock = vi
      .fn()
      .mockResolvedValue(new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);

    const { promise, cleanup } = openOAuthPopup("https://example.com/oauth", {
      stateToken: "tok-server-blocked",
      preOpenedWindow: null,
      cancelUrl: "/api/oauth/pending/cancel",
    });
    promise.catch(() => {});

    cleanup.abort();
    await vi.runAllTicks();

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  test("rejects immediately when window.open is blocked for both attempts", async () => {
    // No preOpenedWindow — the inline open is blocked, and so is the
    // fallback (e.g. aggressive popup blocker). Must not wait for timeout.
    setupPopup(null);

    const { promise, popupBlocked, fallbackBlocked } = openOAuthPopup(
      "https://example.com/oauth",
      {
        stateToken: "tok-blocked-2",
      },
    );
    const onReject = vi.fn();
    promise.catch(onReject);

    await vi.advanceTimersByTimeAsync(10);

    expect(popupBlocked).toBe(true);
    expect(fallbackBlocked).toBe(true);
    expect(onReject).toHaveBeenCalledTimes(1);
    expect((onReject.mock.calls[0][0] as Error).message).toBe(
      OAUTH_ERROR_POPUP_BLOCKED_NO_TAB,
    );
  });
});
