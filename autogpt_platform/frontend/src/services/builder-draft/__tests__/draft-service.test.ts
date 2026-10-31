import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { environment } from "@/services/environment";
import { clearTempFlowId, getOrCreateTempFlowId } from "../draft-service";

const TEMP_ID = /^temp_[0-9a-f-]{36}$/;

// Chrome before 92 and plain-HTTP origins have no crypto.randomUUID.
describe("getOrCreateTempFlowId where crypto.randomUUID is missing", () => {
  beforeEach(() => {
    const originalCrypto = globalThis.crypto;
    vi.stubGlobal("crypto", {
      getRandomValues: originalCrypto.getRandomValues.bind(originalCrypto),
    });
  });

  afterEach(() => {
    clearTempFlowId();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("makes one temp id and keeps it for the session", () => {
    const first = getOrCreateTempFlowId();

    expect(first).toMatch(TEMP_ID);
    expect(getOrCreateTempFlowId()).toBe(first);
  });

  it("makes a temp id on the server", () => {
    vi.spyOn(environment, "isServerSide").mockReturnValue(true);

    expect(getOrCreateTempFlowId()).toMatch(TEMP_ID);
  });
});
