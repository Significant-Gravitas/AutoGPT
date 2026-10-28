import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/services/scripts/scripts", () => ({
  loadScript: vi.fn(),
}));

import { loadScript } from "@/services/scripts/scripts";
import { loadGoogleAPIPicker, loadGoogleIdentityServices } from "../helpers";

const mockLoadScript = loadScript as unknown as ReturnType<typeof vi.fn>;

beforeEach(() => {
  mockLoadScript.mockReset();
  // Simulate an immediate successful script load so the helpers can
  // proceed past the `loadScript` await.
  mockLoadScript.mockResolvedValue(undefined);

  // Provide a minimal stub for the Google global so the downstream
  // checks (`window.gapi`, `window.google`) pass.
  (window as any).gapi = {
    load: (_name: string, opts: { callback: () => void }) => {
      opts.callback();
    },
  };
  (window as any).google = {
    accounts: { oauth2: {} },
  };
});

afterEach(() => {
  delete (window as any).gapi;
  delete (window as any).google;
});

describe("loadGoogleAPIPicker", () => {
  it("loads api.js with the no-referrer-when-downgrade policy", async () => {
    // Firefox respects the default strict-origin-when-cross-origin
    // policy and strips the Referer header on cross-site navigation —
    // Google's api.js issues a picker-token request that fails without
    // a Referer.  Passing `referrerPolicy: "no-referrer-when-downgrade"`
    // keeps the header intact for cross-site same-scheme requests.
    // Pin that policy so a future cleanup doesn't silently drop it.
    await loadGoogleAPIPicker();

    expect(mockLoadScript).toHaveBeenCalledTimes(1);
    const [url, opts] = mockLoadScript.mock.calls[0];
    expect(url).toBe("https://apis.google.com/js/api.js");
    expect(opts).toEqual({ referrerPolicy: "no-referrer-when-downgrade" });
  });

  it("throws if window.gapi is missing after the script loads", async () => {
    delete (window as any).gapi;

    await expect(loadGoogleAPIPicker()).rejects.toThrow(/Google AIP/);
  });
});

describe("loadGoogleIdentityServices", () => {
  it("loads gsi/client with the no-referrer-when-downgrade policy", async () => {
    await loadGoogleIdentityServices();

    expect(mockLoadScript).toHaveBeenCalledTimes(1);
    const [url, opts] = mockLoadScript.mock.calls[0];
    expect(url).toBe("https://accounts.google.com/gsi/client");
    expect(opts).toEqual({ referrerPolicy: "no-referrer-when-downgrade" });
  });

  it("throws if google.accounts.oauth2 is missing after the script loads", async () => {
    (window as any).google = {};

    await expect(loadGoogleIdentityServices()).rejects.toThrow(
      /Google Identity Services not available/,
    );
  });
});
