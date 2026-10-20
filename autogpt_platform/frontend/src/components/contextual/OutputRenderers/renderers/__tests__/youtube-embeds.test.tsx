import { afterEach, describe, expect, it, vi } from "vitest";
import {
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";
import { render, screen } from "@/tests/integrations/test-utils";
import { markdownRenderer } from "../MarkdownRenderer";
import { videoRenderer } from "../VideoRenderer";

const VIDEO_ID = "dQw4w9WgXcQ";
const WATCH_URL = `https://www.youtube.com/watch?v=${VIDEO_ID}`;
const NOCOOKIE_URL = `https://www.youtube-nocookie.com/embed/${VIDEO_ID}`;

function renderMarkdown(markdown: string) {
  return render(
    <>{markdownRenderer.render(markdown, { mimeType: "text/markdown" })}</>,
  ).container;
}

function renderVideo(url: string) {
  return render(<>{videoRenderer.render(url)}</>).container;
}

afterEach(() => {
  removeCookiebot();
  vi.unstubAllEnvs();
});

describe("YouTube links in run output", () => {
  it.each([
    ["markdown", () => renderMarkdown(`# Demo\n\n![Demo](${WATCH_URL})`)],
    ["video", () => renderVideo(`https://youtu.be/${VIDEO_ID}`)],
  ])(
    "shows the cookie notice in %s output without marketing consent",
    (_name, renderOutput) => {
      configureCookiebot();
      installCookiebot({ marketing: false });

      const container = renderOutput();

      expect(container.querySelector("iframe")).toBeNull();
      expect(
        screen.getByRole("link", { name: "Watch on YouTube" }),
      ).toBeDefined();
    },
  );

  it.each([
    ["markdown", () => renderMarkdown(`# Demo\n\n![Demo](${WATCH_URL})`)],
    ["video", () => renderVideo(`https://youtu.be/${VIDEO_ID}`)],
  ])(
    "embeds the privacy-enhanced player in %s output with consent",
    (_name, renderOutput) => {
      configureCookiebot();
      installCookiebot({ marketing: true });

      const container = renderOutput();

      expect(container.querySelector("iframe")?.getAttribute("src")).toBe(
        NOCOOKIE_URL,
      );
    },
  );

  it("drops the share tracking from a youtu.be link in markdown", () => {
    configureCookiebot();
    installCookiebot({ marketing: true });

    const container = renderMarkdown(
      `# Demo\n\n![Demo](https://youtu.be/${VIDEO_ID}?si=abc123)`,
    );

    expect(container.querySelector("iframe")?.getAttribute("src")).toBe(
      NOCOOKIE_URL,
    );
  });
});
