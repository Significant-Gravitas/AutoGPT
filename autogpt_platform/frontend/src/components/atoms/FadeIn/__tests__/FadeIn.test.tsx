import { render, screen } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";

const { motionPreference } = vi.hoisted(() => ({
  motionPreference: { reduced: false },
}));

vi.mock("framer-motion", async (importOriginal) => ({
  ...(await importOriginal<typeof import("framer-motion")>()),
  useReducedMotion: () => motionPreference.reduced,
}));

import { FadeIn } from "../FadeIn";

function fadeIn() {
  const element = screen.getByText("Hello").closest("[data-slot='fade-in']");
  if (!(element instanceof HTMLElement)) throw new Error("no fade-in");
  return element;
}

describe("FadeIn", () => {
  afterEach(() => {
    motionPreference.reduced = false;
  });

  it("rises into place as it fades in", () => {
    render(<FadeIn>Hello</FadeIn>);

    expect(fadeIn().style.opacity).toBe("0");
    expect(fadeIn().style.transform).toContain("translateY(12px)");
  });

  it("only fades in when the user asks for reduced motion", () => {
    motionPreference.reduced = true;
    render(<FadeIn>Hello</FadeIn>);

    expect(fadeIn().style.opacity).toBe("0");
    expect(fadeIn().style.transform).toBe("");
  });
});
