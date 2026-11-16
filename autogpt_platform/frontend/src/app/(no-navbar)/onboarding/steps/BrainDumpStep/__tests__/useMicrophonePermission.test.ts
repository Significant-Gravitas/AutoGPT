import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useMicrophonePermission } from "../useMicrophonePermission";

class FakePermissionStatus extends EventTarget {
  constructor(public state: PermissionState) {
    super();
  }

  change(state: PermissionState) {
    this.state = state;
    this.dispatchEvent(new Event("change"));
  }
}

function stubPermissions(permissions: unknown) {
  vi.stubGlobal("navigator", { ...navigator, permissions });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("useMicrophonePermission", () => {
  it("says unknown when the browser has no Permissions API", async () => {
    stubPermissions(undefined);

    const { result } = renderHook(() => useMicrophonePermission());

    await act(async () => {
      await Promise.resolve();
    });
    expect(result.current.state).toBe("unknown");
  });

  it("says unknown when the browser cannot be asked about the microphone", async () => {
    // Firefox rejects "microphone" as a permission name.
    const query = vi.fn().mockRejectedValue(new TypeError("bad name"));
    stubPermissions({ query });

    const { result } = renderHook(() => useMicrophonePermission());

    await waitFor(() => expect(query).toHaveBeenCalled());
    expect(result.current.state).toBe("unknown");
  });

  it("says unknown when the query answers nothing", async () => {
    const query = vi.fn().mockResolvedValue(undefined);
    stubPermissions({ query });

    const { result } = renderHook(() => useMicrophonePermission());

    await waitFor(() => expect(query).toHaveBeenCalled());
    expect(result.current.state).toBe("unknown");
  });

  it.each(["prompt", "granted", "denied"] as const)(
    "reports a %s microphone",
    async (state) => {
      const query = vi.fn().mockResolvedValue(new FakePermissionStatus(state));
      stubPermissions({ query });

      const { result } = renderHook(() => useMicrophonePermission());

      await waitFor(() => expect(result.current.state).toBe(state));
      expect(query).toHaveBeenCalledWith({ name: "microphone" });
    },
  );

  it("follows a change made in the site settings and says so", async () => {
    const status = new FakePermissionStatus("denied");
    stubPermissions({ query: vi.fn().mockResolvedValue(status) });
    const onChange = vi.fn();

    const { result } = renderHook(() => useMicrophonePermission({ onChange }));
    await waitFor(() => expect(result.current.state).toBe("denied"));

    act(() => {
      status.change("granted");
    });

    expect(result.current.state).toBe("granted");
    expect(onChange).toHaveBeenCalledWith("granted");
  });

  it("reads the state again on request", async () => {
    const status = new FakePermissionStatus("prompt");
    stubPermissions({ query: vi.fn().mockResolvedValue(status) });
    const { result } = renderHook(() => useMicrophonePermission());
    await waitFor(() => expect(result.current.state).toBe("prompt"));

    // A browser that changes the state without firing `change`.
    status.state = "denied";
    let refreshed: string | undefined;
    await act(async () => {
      refreshed = await result.current.refresh();
    });

    expect(refreshed).toBe("denied");
    expect(result.current.state).toBe("denied");
  });

  it("stops listening when unmounted", async () => {
    const status = new FakePermissionStatus("prompt");
    const removeListener = vi.spyOn(status, "removeEventListener");
    stubPermissions({ query: vi.fn().mockResolvedValue(status) });
    const onChange = vi.fn();

    const { result, unmount } = renderHook(() =>
      useMicrophonePermission({ onChange }),
    );
    await waitFor(() => expect(result.current.state).toBe("prompt"));
    unmount();
    status.change("granted");

    expect(removeListener).toHaveBeenCalledWith("change", expect.any(Function));
    expect(onChange).not.toHaveBeenCalled();
  });
});
