import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { ExpertAvatar } from "./ExpertAvatar";

describe("ExpertAvatar saved appearance", () => {
  test("uses the saved clay identity after a rename and color change", () => {
    const avatarUrl = "/autogpt-characters/v1.1/expert-mina/neutral/128.webp";
    const { rerender } = render(
      <ExpertAvatar name="Mina" avatarUrl={avatarUrl} size={32} />,
    );
    const image = screen.getByRole("img");
    expect(image.getAttribute("src")).toBe(
      "/autogpt-characters/v1.1/expert-mina/neutral/32.webp",
    );
    rerender(
      <ExpertAvatar
        name="My editor"
        avatarUrl={avatarUrl}
        color="green-300"
        size={32}
      />,
    );
    expect(screen.getByRole("img").getAttribute("src")).toBe(
      image.getAttribute("src"),
    );
  });

  test("falls back to PNG then initials without borrowing another identity", () => {
    render(
      <ExpertAvatar
        name="My editor"
        avatarUrl="/autogpt-characters/v1.1/expert-mina/neutral/128.webp"
        size={32}
      />,
    );
    fireEvent.error(screen.getByRole("img"));
    expect(screen.getByRole("img").getAttribute("src")).toBe(
      "/autogpt-characters/v1.1/expert-mina/neutral/32.png",
    );
    fireEvent.error(screen.getByRole("img"));
    expect(screen.getByRole("img").textContent).toBe("MY");
  });

  test("does not replace a custom avatar based on its owner's name", () => {
    const { container } = render(
      <ExpertAvatar
        name="Maria"
        avatarUrl="https://example.com/my-image.png"
      />,
    );
    expect(container.querySelector('[src*="autogpt-characters"]')).toBeNull();
  });
});

test("uses transparent managed artwork on a topic background", () => {
  render(
    <ExpertAvatar
      name="Mina"
      avatarUrl="/autogpt-characters/v1.1/expert-mina/neutral/128.webp"
      backgroundColor="#A5B09A"
      size={88}
    />,
  );
  expect(screen.getByRole("img").getAttribute("src")).toContain(
    "/experts/transparent/mina.webp",
  );
});
