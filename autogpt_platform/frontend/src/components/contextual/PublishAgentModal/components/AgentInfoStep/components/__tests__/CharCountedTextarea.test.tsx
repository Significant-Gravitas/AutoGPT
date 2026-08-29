import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { CharCountedTextarea } from "../CharCountedTextarea";

describe("CharCountedTextarea", () => {
  it("renders the children and a length / max badge", () => {
    render(
      <CharCountedTextarea max={100} value="hello">
        <textarea />
      </CharCountedTextarea>,
    );

    const badge = screen.getByTestId("char-count");
    expect(badge.textContent).toBe("5 / 100");
    expect(badge.className).toMatch(/text-zinc-400/);
    expect(screen.getByRole("textbox")).toBeDefined();
  });

  it("turns rose-600 when the value exceeds the max", () => {
    render(
      <CharCountedTextarea max={3} value="too long">
        <textarea />
      </CharCountedTextarea>,
    );

    const badge = screen.getByTestId("char-count");
    expect(badge.textContent).toBe("8 / 3");
    expect(badge.className).toMatch(/text-rose-600/);
  });
});
