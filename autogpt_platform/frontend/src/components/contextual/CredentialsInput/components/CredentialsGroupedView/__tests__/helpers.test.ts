import { describe, expect, it } from "vitest";

import { findSavedUserCredentialByProviderAndType } from "../helpers";

function providersWithHost(host: string) {
  return {
    http: {
      savedCredentials: [
        { id: "cred-1", provider: "http", type: "host_scoped", host },
      ],
    },
  } as never;
}

function findFor(host: string, url: string) {
  return findSavedUserCredentialByProviderAndType(
    ["http"],
    ["host_scoped"],
    undefined,
    providersWithHost(host),
    [url],
  );
}

describe("findSavedUserCredentialByProviderAndType, host-scoped matching", () => {
  it("offers a credential whose host names the URL's non-default port", () => {
    expect(
      findFor("api.example.com:8443", "https://api.example.com:8443/v1")?.id,
    ).toBe("cred-1");
    expect(
      findFor("api.example.com:8443", "HTTPS://api.example.com:8443/v1")?.id,
    ).toBe("cred-1");
  });

  // The backend refuses a request port outside 80/443 unless the credential
  // names it, so offering either of these would save headers that never send.
  it("does not offer one whose port disagrees with the URL", () => {
    expect(
      findFor("api.example.com", "https://api.example.com:8443/v1"),
    ).toBeUndefined();
    expect(
      findFor("api.example.com:8443", "https://api.example.com/v1"),
    ).toBeUndefined();
  });

  it("still matches when neither names a port", () => {
    expect(findFor("api.example.com", "https://api.example.com/v1")?.id).toBe(
      "cred-1",
    );
    expect(
      findFor("api.example.com", "https://api.example.com:443/v1")?.id,
    ).toBe("cred-1");
  });
});
