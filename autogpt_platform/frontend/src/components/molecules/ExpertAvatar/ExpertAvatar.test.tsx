import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { ExpertAvatar } from "./ExpertAvatar";

const MINA = "/autogpt-characters/v1.1/expert-mina/neutral/128.webp";
const JULES = "/autogpt-characters/v2.1/expert-jules/neutral/128.webp";

describe("ExpertAvatar saved appearance", () => {
  test("uses the saved clay identity after a rename and color change", () => {
    const { rerender } = render(
      <ExpertAvatar name="Mina" avatarUrl={MINA} size={32} />,
    );
    const image = screen.getByRole("img");
    expect(image.getAttribute("src")).toBe(
      "/autogpt-characters/v1.1/expert-mina/neutral/32.webp",
    );
    rerender(
      <ExpertAvatar
        name="My editor"
        avatarUrl={MINA}
        color="green-300"
        size={32}
      />,
    );
    expect(screen.getByRole("img").getAttribute("src")).toBe(
      image.getAttribute("src"),
    );
  });

  test("serves a 2.1 identity from its own library with 1x and 2x sources", () => {
    const { container } = render(
      <ExpertAvatar name="Jules" avatarUrl={JULES} size={88} />,
    );
    expect(screen.getByRole("img").getAttribute("src")).toBe(
      "/autogpt-characters/v2.1/expert-jules/neutral/96.webp",
    );
    expect(container.querySelector("source")?.getAttribute("srcset")).toBe(
      "/autogpt-characters/v2.1/expert-jules/neutral/96.webp 1x, /autogpt-characters/v2.1/expert-jules/neutral/192.webp 2x",
    );
  });

  test("a retired clay default shows the identity it stood for, not the filter's variant", () => {
    render(
      <ExpertAvatar
        name="Jules"
        avatarUrl="/experts/clay/v4/jules-content.png"
        size={40}
      />,
    );
    expect(screen.getByRole("img").getAttribute("src")).toBe(
      "/autogpt-characters/v2.1/expert-jules/neutral/40.webp",
    );
  });

  test("falls back to PNG then initials without borrowing another identity", () => {
    render(<ExpertAvatar name="My editor" avatarUrl={MINA} size={32} />);
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

  test("a missing avatar shows the General fallback, never Otto", () => {
    render(<ExpertAvatar name="Nova" avatarUrl={null} size={40} />);
    expect(screen.getByRole("img").getAttribute("src")).toBe(
      "/autogpt-characters/v2.1/expert-general-01/neutral/40.webp",
    );
  });
});

test("a managed identity keeps its opaque studio tile on a topic background", () => {
  render(
    <ExpertAvatar
      name="Mina"
      avatarUrl={MINA}
      backgroundColor="#A5B09A"
      size={88}
    />,
  );
  const image = screen.getByRole("img");
  expect(image.getAttribute("src")).toBe(
    "/autogpt-characters/v1.1/expert-mina/neutral/96.webp",
  );
  expect(image.getAttribute("alt")).toBe("Mina, AI Expert");
});
