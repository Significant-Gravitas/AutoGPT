import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { describe, expect, test } from "vitest";
import { InitialAvatar } from "../InitialAvatar";

function getFallbackSVG(container: HTMLElement) {
  const svg = container.querySelector("svg");
  return svg?.innerHTML.replace(/:r[0-9a-z]+:/g, "id") ?? null;
}

describe("InitialAvatar", () => {
  test("renders marble gradient fallback when no image is provided", () => {
    const { container } = render(<InitialAvatar name="abhimanyu" />);
    expect(getFallbackSVG(container)).not.toBeNull();
  });

  test("seeds the gradient deterministically by name", () => {
    const first = render(<InitialAvatar name="ada" />);
    const second = render(<InitialAvatar name="ada" />);
    const other = render(<InitialAvatar name="beth" />);

    expect(getFallbackSVG(first.container)).toBe(
      getFallbackSVG(second.container),
    );
    expect(getFallbackSVG(first.container)).not.toBe(
      getFallbackSVG(other.container),
    );
  });

  test("trims the name before seeding the gradient", () => {
    const padded = render(<InitialAvatar name="   beth" />);
    const plain = render(<InitialAvatar name="beth" />);
    expect(getFallbackSVG(padded.container)).toBe(
      getFallbackSVG(plain.container),
    );
  });

  test("falls back to the 'User' seed when name is missing or empty", () => {
    const missing = render(<InitialAvatar />);
    const empty = render(<InitialAvatar name="" />);
    const user = render(<InitialAvatar name="User" />);

    expect(getFallbackSVG(missing.container)).toBe(
      getFallbackSVG(user.container),
    );
    expect(getFallbackSVG(empty.container)).toBe(
      getFallbackSVG(user.container),
    );
  });

  test("prefers username over display name as the gradient seed", () => {
    const withUsername = render(
      <InitialAvatar name="Ada Lovelace" username="ada" />,
    );
    const usernameOnly = render(<InitialAvatar name="ada" />);
    const nameOnly = render(<InitialAvatar name="Ada Lovelace" />);

    expect(getFallbackSVG(withUsername.container)).toBe(
      getFallbackSVG(usernameOnly.container),
    );
    expect(getFallbackSVG(withUsername.container)).not.toBe(
      getFallbackSVG(nameOnly.container),
    );
  });

  test("merges className prop into the avatar root", () => {
    const { container } = render(
      <InitialAvatar name="ada" className="size-12" />,
    );
    const root = container.firstChild as HTMLElement;
    expect(root.className).toContain("size-12");
  });

  test("shows a decorative image when src is provided", async () => {
    const { container } = render(
      <InitialAvatar name="ada" src="https://example.com/avatar.png" />,
    );

    await waitFor(() => {
      const image = container.querySelector("img");
      expect(image).not.toBeNull();
      expect(image?.getAttribute("aria-hidden")).toBe("true");
    });
    expect(screen.queryByRole("img")).toBeNull();
  });
});
