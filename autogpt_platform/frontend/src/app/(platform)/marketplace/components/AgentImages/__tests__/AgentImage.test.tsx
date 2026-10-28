import { afterEach, describe, expect, test, vi } from "vitest";
import {
  act,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import {
  answerCookiebot,
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";

import { AgentImages } from "../AgentImage";

const IMAGES = ["https://cdn.test/one.png", "https://cdn.test/two.png"];

describe("AgentImages thumbnails", () => {
  test("renders a thumbnail per image", () => {
    render(<AgentImages images={IMAGES} />);
    expect(screen.getByAltText("Thumbnail 1")).toBeDefined();
    expect(screen.getByAltText("Thumbnail 2")).toBeDefined();
  });

  test("replaces only the dead thumbnail with a placeholder", () => {
    render(<AgentImages images={IMAGES} />);

    fireEvent.error(screen.getByAltText("Thumbnail 2"));

    expect(screen.queryByAltText("Thumbnail 2")).toBeNull();
    expect(screen.getByAltText("Thumbnail 1")).toBeDefined();
  });

  test("shows a replacement image at a previously dead index", () => {
    const { rerender } = render(<AgentImages images={IMAGES} />);
    fireEvent.error(screen.getByAltText("Thumbnail 2"));

    rerender(
      <AgentImages images={[IMAGES[0], "https://cdn.test/replacement.png"]} />,
    );

    const replacement = screen.getByAltText("Thumbnail 2") as HTMLImageElement;
    expect(replacement.src).toBe("https://cdn.test/replacement.png");
  });
});

describe("AgentImages YouTube thumbnails", () => {
  const WITH_SHORT = [
    "https://cdn.test/one.png",
    "https://www.youtube.com/shorts/dQw4w9WgXcQ",
  ];

  afterEach(() => {
    removeCookiebot();
    vi.unstubAllEnvs();
  });

  test("loads nothing from YouTube before marketing consent", () => {
    configureCookiebot();
    installCookiebot({ marketing: false });

    const { container } = render(<AgentImages images={WITH_SHORT} />);

    expect(screen.queryByAltText("Thumbnail 2")).toBeNull();
    expect(screen.getByText("YouTube video")).toBeDefined();
    const requested = Array.from(container.querySelectorAll("img, iframe"))
      .map((node) => node.getAttribute("src") ?? "")
      .filter((src) => src.includes("youtube"));
    expect(requested).toEqual([]);
  });

  test("shows the YouTube thumbnail once marketing cookies are allowed", () => {
    configureCookiebot();
    installCookiebot({ marketing: false });
    render(<AgentImages images={WITH_SHORT} />);

    act(() => answerCookiebot({ marketing: true }));

    expect(screen.getByAltText("Thumbnail 2").getAttribute("src")).toBe(
      "https://img.youtube.com/vi/dQw4w9WgXcQ/mqdefault.jpg",
    );
  });

  test.each([
    "https://img.youtube.com/vi/dQw4w9WgXcQ/maxresdefault.jpg",
    "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg",
    "https://i.ytimg.com./vi/dQw4w9WgXcQ/mqdefault.jpg",
    "https://img.youtube.com./vi/dQw4w9WgXcQ/0.jpg",
  ])("waits for consent before a listing image hosted at %s", (image) => {
    configureCookiebot();
    installCookiebot({ marketing: false });
    const { container } = render(
      <AgentImages images={["https://cdn.test/one.png", image]} />,
    );

    expect(screen.queryByAltText("Thumbnail 2")).toBeNull();
    expect(screen.getByText("Image from YouTube")).toBeDefined();
    const requested = Array.from(container.querySelectorAll("img"))
      .map((node) => node.getAttribute("src") ?? "")
      .filter((src) => src.includes("youtube") || src.includes("ytimg"));
    expect(requested).toEqual([]);

    act(() => answerCookiebot({ marketing: true }));

    expect(screen.getByAltText("Thumbnail 2").getAttribute("src")).toBe(image);
  });
});
