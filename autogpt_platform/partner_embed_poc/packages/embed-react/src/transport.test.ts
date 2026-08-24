import { afterEach, describe, expect, it, vi } from "vitest";

import { createMessageID } from "./transport";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("createMessageID", () => {
  it("uses randomUUID when the browser provides it", () => {
    const randomUUID = vi
      .fn()
      .mockReturnValue("c14a3e0f-68a6-401e-8ad8-070bf84e05f2");
    vi.stubGlobal("crypto", { randomUUID } satisfies Partial<Crypto>);

    expect(createMessageID()).toBe("c14a3e0f-68a6-401e-8ad8-070bf84e05f2");
    expect(randomUUID).toHaveBeenCalledOnce();
  });

  it("creates an RFC 4122 UUID when randomUUID is unavailable", () => {
    function getRandomValues<T extends ArrayBufferView | null>(array: T): T {
      if (array instanceof Uint8Array) array.fill(0);
      return array;
    }
    vi.stubGlobal("crypto", { getRandomValues } satisfies Partial<Crypto>);

    expect(createMessageID()).toBe("00000000-0000-4000-8000-000000000000");
  });

  it("omits the optional message ID when Web Crypto is unavailable", () => {
    vi.stubGlobal("crypto", undefined);

    expect(createMessageID()).toBeUndefined();
  });
});
