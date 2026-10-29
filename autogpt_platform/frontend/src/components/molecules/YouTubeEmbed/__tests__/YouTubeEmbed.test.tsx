import { afterEach, describe, expect, test, vi } from "vitest";
import {
  answerCookiebot,
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";
import {
  act,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import { YouTubeEmbed } from "../YouTubeEmbed";

const VIDEO_ID = "dQw4w9WgXcQ";
const VIDEO_URL = `https://www.youtube.com/watch?v=${VIDEO_ID}`;
const NOTICE = "YouTube sets advertising cookies when this video loads.";

function renderEmbed() {
  return render(<YouTubeEmbed url={VIDEO_URL} title="Demo video" />);
}

function getPlayer(container: HTMLElement) {
  return container.querySelector("iframe");
}

afterEach(() => {
  removeCookiebot();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("YouTubeEmbed", () => {
  test("loads the privacy-enhanced player once marketing cookies are allowed", () => {
    configureCookiebot();
    installCookiebot({ marketing: true });

    const { container } = renderEmbed();

    const player = getPlayer(container);
    expect(player?.getAttribute("src")).toBe(
      `https://www.youtube-nocookie.com/embed/${VIDEO_ID}`,
    );
    expect(player?.getAttribute("title")).toBe("Demo video");
    expect(player?.hasAttribute("allowfullscreen")).toBe(true);
    expect(player?.getAttribute("sandbox")).toBe(
      "allow-scripts allow-same-origin allow-presentation allow-popups allow-popups-to-escape-sandbox",
    );
    expect(screen.queryByText(NOTICE, { exact: false })).toBeNull();
  });

  test("shows a notice instead of the player without marketing consent", () => {
    configureCookiebot();
    installCookiebot({ statistics: true, marketing: false });

    const { container } = renderEmbed();

    expect(getPlayer(container)).toBeNull();
    expect(
      screen.getByText(
        "YouTube sets advertising cookies when this video loads. To watch it here, allow marketing cookies.",
      ),
    ).toBeDefined();
    const watchLink = screen.getByRole("link", { name: "Watch on YouTube" });
    expect(watchLink.getAttribute("href")).toBe(
      `https://www.youtube.com/watch?v=${VIDEO_ID}`,
    );
    expect(watchLink.getAttribute("target")).toBe("_blank");
    expect(watchLink.getAttribute("rel")).toContain("noopener");
  });

  test("opens the cookie settings from the notice", () => {
    configureCookiebot();
    const { renew } = installCookiebot({ marketing: false });

    renderEmbed();
    fireEvent.click(screen.getByRole("button", { name: "Cookie settings" }));

    expect(renew).toHaveBeenCalledOnce();
  });

  test("plays without a reload when the visitor allows marketing cookies", () => {
    configureCookiebot();
    installCookiebot({ marketing: false });
    const { container } = renderEmbed();
    expect(getPlayer(container)).toBeNull();

    act(() => answerCookiebot({ statistics: true, marketing: true }));

    expect(getPlayer(container)?.getAttribute("src")).toBe(
      `https://www.youtube-nocookie.com/embed/${VIDEO_ID}`,
    );
  });

  test("removes the player when the visitor takes marketing consent back", () => {
    configureCookiebot();
    installCookiebot({ marketing: true });
    const { container } = renderEmbed();
    expect(getPlayer(container)).not.toBeNull();

    act(() => answerCookiebot({ statistics: true, marketing: false }));

    expect(getPlayer(container)).toBeNull();
    expect(
      screen.getByRole("link", { name: "Watch on YouTube" }),
    ).toBeDefined();
  });

  test("keeps the settings button disabled until the consent banner loads", () => {
    configureCookiebot();
    vi.spyOn(document, "readyState", "get").mockReturnValue("loading");

    const { container } = renderEmbed();

    expect(getPlayer(container)).toBeNull();
    const settings = screen.getByRole("button", { name: "Cookie settings" });
    expect(settings.hasAttribute("disabled")).toBe(true);
  });

  test("lets the viewer load one video when there is no consent banner", () => {
    vi.stubEnv("NEXT_PUBLIC_COOKIEBOT_CBID", "");

    const { container } = renderEmbed();

    expect(getPlayer(container)).toBeNull();
    expect(screen.getByText(NOTICE)).toBeDefined();
    expect(
      screen.queryByRole("button", { name: "Cookie settings" }),
    ).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Load video" }));

    expect(getPlayer(container)?.getAttribute("src")).toBe(
      `https://www.youtube-nocookie.com/embed/${VIDEO_ID}`,
    );
  });

  test("asks again before loading a different video without a banner", () => {
    vi.stubEnv("NEXT_PUBLIC_COOKIEBOT_CBID", "");
    const { container, rerender } = renderEmbed();
    fireEvent.click(screen.getByRole("button", { name: "Load video" }));
    expect(getPlayer(container)).not.toBeNull();

    rerender(
      <YouTubeEmbed url="https://youtu.be/aaaaaaaaaaa" title="Other video" />,
    );

    expect(getPlayer(container)).toBeNull();
    expect(screen.getByRole("button", { name: "Load video" })).toBeDefined();
  });

  test("only links to YouTube when the consent banner could not load", () => {
    configureCookiebot();
    vi.spyOn(document, "readyState", "get").mockReturnValue("complete");

    const { container } = renderEmbed();

    expect(getPlayer(container)).toBeNull();
    expect(screen.getByText(NOTICE)).toBeDefined();
    expect(
      screen.getByRole("link", { name: "Watch on YouTube" }),
    ).toBeDefined();
    expect(
      screen.queryByRole("button", { name: "Cookie settings" }),
    ).toBeNull();
    expect(screen.queryByRole("button", { name: "Load video" })).toBeNull();
  });

  test("only links out for a YouTube link that names no single video", () => {
    configureCookiebot();
    installCookiebot({ marketing: true });

    const { container } = render(
      <YouTubeEmbed url="https://www.youtube.com/@somechannel" />,
    );

    expect(getPlayer(container)).toBeNull();
    expect(
      screen.getByText("This video can only be watched on YouTube."),
    ).toBeDefined();
    expect(
      screen
        .getByRole("link", { name: "Watch on YouTube" })
        .getAttribute("href"),
    ).toBe("https://www.youtube.com/@somechannel");
    expect(
      screen.queryByRole("button", { name: "Cookie settings" }),
    ).toBeNull();
  });
});
