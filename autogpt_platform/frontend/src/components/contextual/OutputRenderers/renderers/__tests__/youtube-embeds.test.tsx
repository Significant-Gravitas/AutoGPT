import { afterEach, describe, expect, it, vi } from "vitest";
import {
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
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

function renderVideo(url: string, metadata?: { type: "video" }) {
  return render(<>{videoRenderer.render(url, metadata)}</>).container;
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

  it("keeps a YouTube thumbnail in markdown as an image", () => {
    const thumbnail = `https://img.youtube.com/vi/${VIDEO_ID}/0.jpg`;

    const container = renderMarkdown(`# Demo\n\n![Thumbnail](${thumbnail})`);

    expect(container.querySelector("img")?.getAttribute("src")).toBe(thumbnail);
    expect(screen.queryByRole("link", { name: "Watch on YouTube" })).toBeNull();
  });

  it("keeps a YouTube channel link in markdown as a plain link", () => {
    configureCookiebot();
    installCookiebot({ marketing: true });
    const channel = "https://www.youtube.com/@somechannel";

    const container = renderMarkdown(
      `# Demo\n\nSee [the channel](${channel}) and ![Channel](${channel})`,
    );

    const links = screen.getAllByRole("link", { name: /channel/i });
    expect(links.map((link) => link.getAttribute("href"))).toEqual([
      channel,
      channel,
    ]);
    expect(container.querySelector("img, video, iframe")).toBeNull();
    expect(
      screen.queryByText("This video can only be watched on YouTube."),
    ).toBeNull();
  });

  it("links a YouTube channel given as video output instead of playing it", () => {
    configureCookiebot();
    installCookiebot({ marketing: true });
    const channel = "https://www.youtube.com/@somechannel";

    expect(videoRenderer.canRender(channel)).toBe(false);
    const container = renderVideo(channel, { type: "video" });

    expect(container.querySelector("video, iframe, img")).toBeNull();
    expect(screen.getByRole("link", { name: channel })).toBeDefined();
  });

  it("keeps one link for a linked image of a YouTube channel", () => {
    configureCookiebot();
    installCookiebot({ marketing: true });
    const channel = "https://www.youtube.com/@somechannel";

    const container = renderMarkdown(
      `# Demo\n\n[![Channel](${channel})](${channel})`,
    );

    const anchors = container.querySelectorAll(`a[href="${channel}"]`);
    expect(anchors).toHaveLength(1);
    expect(anchors[0].getAttribute("href")).toBe(channel);
    expect(anchors[0].textContent).toBe("Channel");
    expect(container.querySelector("a a, img, video, iframe")).toBeNull();
  });

  it("shows only the text of a linked YouTube video image", () => {
    configureCookiebot();
    installCookiebot({ marketing: false });

    const container = renderMarkdown(
      `# Demo\n\n[![Watch the demo](${WATCH_URL})](https://example.com/page)`,
    );

    const anchor = container.querySelector(
      'a[href="https://example.com/page"]',
    );
    expect(anchor?.textContent).toBe("Watch the demo");
    expect(anchor?.querySelector("a, button, iframe, video, img")).toBeNull();
  });

  it("gives a YouTube channel image written without a scheme a full link", () => {
    const container = renderMarkdown(`# Demo\n\n![Channel](youtube.com/@foo)`);

    expect(
      container.querySelector('a[href="https://youtube.com/@foo"]')
        ?.textContent,
    ).toBe("Channel");
    expect(container.querySelector("img")).toBeNull();
  });

  describe("when the output around the player re-renders", () => {
    const markdown = `# Demo\n\n![Demo](${WATCH_URL})`;

    function renderInParent(pass: number) {
      return (
        <div data-pass={pass}>
          {markdownRenderer.render(markdown, { mimeType: "text/markdown" })}
        </div>
      );
    }

    it("keeps the same player element", () => {
      configureCookiebot();
      installCookiebot({ marketing: true });
      const { container, rerender } = render(renderInParent(1));
      const player = container.querySelector("iframe");
      expect(player).not.toBeNull();

      rerender(renderInParent(2));

      expect(container.querySelector("iframe")).toBe(player);
    });

    it("keeps a video loaded with Load video", () => {
      vi.stubEnv("NEXT_PUBLIC_COOKIEBOT_CBID", "");
      const { container, rerender } = render(renderInParent(1));
      fireEvent.click(screen.getByRole("button", { name: "Load video" }));
      const player = container.querySelector("iframe");
      expect(player).not.toBeNull();

      rerender(renderInParent(2));

      expect(container.querySelector("iframe")).toBe(player);
    });
  });
});
