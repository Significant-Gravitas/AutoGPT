import { describe, expect, it } from "vitest";
import {
  isMicrophonePermissionDenied,
  microphoneErrorName,
} from "./microphone";

describe("isMicrophonePermissionDenied", () => {
  it("recognises the browser's refusal", () => {
    expect(
      isMicrophonePermissionDenied(new DOMException("no", "NotAllowedError")),
    ).toBe(true);
  });

  // A missing microphone or a generic failure is not a denial — treating
  // it as one would send the user to the typed fallback for good.
  it("does not treat other failures as a denial", () => {
    expect(
      isMicrophonePermissionDenied(new DOMException("gone", "NotFoundError")),
    ).toBe(false);
    expect(isMicrophonePermissionDenied(new Error("NotAllowedError"))).toBe(
      false,
    );
    expect(isMicrophonePermissionDenied(null)).toBe(false);
  });
});

describe("microphoneErrorName", () => {
  it("names a browser failure, and says so when there is nothing to name", () => {
    expect(
      microphoneErrorName(new DOMException("busy", "NotReadableError")),
    ).toBe("NotReadableError");
    expect(microphoneErrorName(new TypeError("no mediaDevices"))).toBe(
      "TypeError",
    );
    expect(microphoneErrorName("nope")).toBe("unknown");
  });
});
