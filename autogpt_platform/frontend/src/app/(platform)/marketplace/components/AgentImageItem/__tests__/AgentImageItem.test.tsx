import { afterEach, describe, expect, test, vi } from "vitest";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import {
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";

import { AgentImageItem } from "../AgentImageItem";

function renderItem(image: string) {
  return render(
    <AgentImageItem
      image={image}
      index={0}
      playingVideoIndex={null}
      handlePlay={() => {}}
      handlePause={() => {}}
    />,
  );
}

describe("AgentImageItem", () => {
  afterEach(() => {
    removeCookiebot();
    vi.unstubAllEnvs();
  });

  test("renders the preview image while the URL loads", () => {
    renderItem("https://cdn.test/preview.png");
    expect(screen.getByAltText("Image")).toBeDefined();
  });

  test("replaces a dead preview URL with a placeholder", () => {
    const { container } = renderItem("https://cdn.test/dead.png");

    fireEvent.error(screen.getByAltText("Image"));

    expect(screen.queryByAltText("Image")).toBeNull();
    expect(container.querySelector("svg")).not.toBeNull();
  });

  test("shows the cookie notice instead of a YouTube video without consent", () => {
    configureCookiebot();
    installCookiebot({ marketing: false });

    const { container } = renderItem(
      "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    );

    expect(container.querySelector("iframe")).toBeNull();
    expect(
      screen.getByRole("link", { name: "Watch on YouTube" }),
    ).toBeDefined();
  });

  test("embeds the privacy-enhanced YouTube player with consent", () => {
    configureCookiebot();
    installCookiebot({ marketing: true });

    const { container } = renderItem(
      "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    );

    expect(container.querySelector("iframe")?.getAttribute("src")).toBe(
      "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ",
    );
  });

  test.each([
    "https://www.youtube.com/shorts/dQw4w9WgXcQ",
    "https://www.youtube.com/live/dQw4w9WgXcQ",
    "https://www.youtube.com/watch?feature=shared&v=dQw4w9WgXcQ",
  ])("gates %s behind consent like any other video", (url) => {
    configureCookiebot();
    installCookiebot({ marketing: false });

    const { container } = renderItem(url);

    expect(container.querySelector("iframe")).toBeNull();
    expect(container.querySelector("video")).toBeNull();
    expect(
      screen
        .getByRole("link", { name: "Watch on YouTube" })
        .getAttribute("href"),
    ).toBe("https://www.youtube.com/watch?v=dQw4w9WgXcQ");
  });

  test("never puts a YouTube page it can't play into a video element", () => {
    configureCookiebot();
    installCookiebot({ marketing: true });

    const { container } = renderItem("https://www.youtube.com/@somechannel");

    expect(container.querySelector("video")).toBeNull();
    expect(container.querySelector("iframe")).toBeNull();
    expect(
      screen.getByRole("link", { name: "Watch on YouTube" }),
    ).toBeDefined();
  });
});
