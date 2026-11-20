import { afterEach, describe, expect, it, vi } from "vitest";
import { removeGtagShim } from "@/tests/integrations/gtag-shim";
import { gtag } from "./gtag";

describe("gtag", () => {
  afterEach(() => {
    removeGtagShim();
  });

  it("forwards the command to the tag's own shim", () => {
    const shim = vi.fn();
    window.gtag = shim;

    expect(gtag("event", "conversion", { send_to: "AW-1/x" })).toBe(true);

    expect(shim).toHaveBeenCalledWith("event", "conversion", {
      send_to: "AW-1/x",
    });
  });

  it("drops the command when the tag has not been set up", () => {
    removeGtagShim();

    expect(gtag("event", "conversion", { send_to: "AW-1/x" })).toBe(false);
  });
});
