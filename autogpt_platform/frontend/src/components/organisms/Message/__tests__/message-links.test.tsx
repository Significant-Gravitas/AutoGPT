import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { remarkAppLinks, toAppPath } from "../app-links";
import { MessageLink } from "../components/MessageLink/MessageLink";
import { MessageResponse } from "../Message";

const AGENT_ID = "144760ff-de7d-4cc2-94ee-4bccaa741cf7";

async function renderReply(markdown: string) {
  const { container } = render(<MessageResponse>{markdown}</MessageResponse>);
  await waitFor(() => expect(container.textContent).not.toBe(""));
  return container;
}

function linksOf(container: HTMLElement) {
  return Array.from(
    container.querySelectorAll<HTMLAnchorElement>('[data-streamdown="link"]'),
  );
}

function middleClick(link: HTMLElement) {
  return fireEvent(
    link,
    new MouseEvent("auxclick", { bubbles: true, cancelable: true, button: 1 }),
  );
}

// Whether the browser would follow the link after the app's handlers ran. The
// test environment would then load the page, so the default is stopped here,
// after React's handlers (they listen on the root, below the window).
function leavesToBrowser(
  link: HTMLElement,
  click: (link: HTMLElement) => void,
) {
  let followed = false;
  function stopNavigation(event: Event) {
    followed = !event.defaultPrevented;
    event.preventDefault();
  }
  window.addEventListener("click", stopNavigation);
  window.addEventListener("auxclick", stopNavigation);
  try {
    click(link);
  } finally {
    window.removeEventListener("click", stopNavigation);
    window.removeEventListener("auxclick", stopNavigation);
  }
  return followed;
}

function hrefOf(link: HTMLElement) {
  return link.getAttribute("href");
}

describe("MessageResponse links", () => {
  beforeEach(() => {
    vi.spyOn(window, "open").mockImplementation(() => null);
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("turns a bare app path into a same-site link", async () => {
    const container = await renderReply(
      `You can [open it in your library](library/agents/${AGENT_ID}) to run it yourself.`,
    );

    const links = linksOf(container);
    expect(links.map((link) => link.textContent)).toEqual([
      "open it in your library",
    ]);
    expect(container.textContent).not.toContain("[blocked]");
    expect(hrefOf(links[0])).toBe(`/library/agents/${AGENT_ID}`);
  });

  it("leaves an app path with a leading slash as it is", async () => {
    const container = await renderReply(
      `[open it](/library/agents/${AGENT_ID})`,
    );

    expect(hrefOf(linksOf(container)[0])).toBe(`/library/agents/${AGENT_ID}`);
  });

  it("opens a same-site link in a new tab without asking", async () => {
    const container = await renderReply(
      `[open it](/library/agents/${AGENT_ID})`,
    );
    const link = screen.getByRole("link", { name: "open it" });

    expect(link.getAttribute("target")).toBe("_blank");
    expect(leavesToBrowser(link, fireEvent.click)).toBe(true);
    expect(screen.queryByText("Open external link")).toBeNull();
    expect(linksOf(container)).toEqual([link]);
  });

  it("asks before a plain click leaves the site", async () => {
    await renderReply("[docs](https://docs.agpt.co/guide)");
    const link = screen.getByRole("link", { name: "docs" });

    expect(hrefOf(link)).toBe("https://docs.agpt.co/guide");
    expect(leavesToBrowser(link, fireEvent.click)).toBe(false);
    expect(await screen.findByText("https://docs.agpt.co/guide")).toBeTruthy();
    expect(window.open).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Open link" }));
    expect(window.open).toHaveBeenCalledWith(
      "https://docs.agpt.co/guide",
      "_blank",
      "noreferrer",
    );
    await waitFor(() =>
      expect(screen.queryByText("Open external link")).toBeNull(),
    );
  });

  it.each([
    ["a middle-click", middleClick],
    [
      "a Ctrl-click",
      (link: HTMLElement) => fireEvent.click(link, { ctrlKey: true }),
    ],
    [
      "a Cmd-click",
      (link: HTMLElement) => fireEvent.click(link, { metaKey: true }),
    ],
    [
      "a Shift-click",
      (link: HTMLElement) => fireEvent.click(link, { shiftKey: true }),
    ],
  ])(
    "leaves %s on a link to another site to the browser",
    async (_, clickLink) => {
      await renderReply("[docs](https://docs.agpt.co/guide)");
      const link = screen.getByRole("link", { name: "docs" });

      expect(leavesToBrowser(link, clickLink)).toBe(true);
      expect(screen.queryByText("Open external link")).toBeNull();
      expect(window.open).not.toHaveBeenCalled();
    },
  );

  it("still blocks a javascript: link", async () => {
    const container = await renderReply("[click](javascript:alert(1))");

    expect(linksOf(container)).toEqual([]);
    expect(container.textContent).toContain("[blocked]");
  });

  it("does not rewrite a bare path outside the app's routes", async () => {
    const container = await renderReply("[site](example.com/x)");

    expect(linksOf(container)).toEqual([]);
    expect(container.textContent).toContain("[blocked]");
  });

  it("leaves the same text inside code untouched", async () => {
    const container = await renderReply(
      "Inline `[a](library/agents/1)` and:\n\n```\n[b](library/agents/2)\n```\n",
    );

    await waitFor(() =>
      expect(container.textContent).toContain("[b](library/agents/2)"),
    );
    expect(container.textContent).toContain("[a](library/agents/1)");
    expect(linksOf(container)).toEqual([]);
  });
});

// Streamdown's link hardening turns a link that is still streaming in into
// "[blocked]" before it reaches the renderer; this covers the renderer alone.
describe("MessageLink", () => {
  afterEach(cleanup);

  it("keeps Streamdown's placeholder for a link still streaming in inert", () => {
    render(
      <MessageLink href="streamdown:incomplete-link">the docs</MessageLink>,
    );
    const link = screen.getByText("the docs");

    expect(link.hasAttribute("href")).toBe(false);
    fireEvent.click(link);
    expect(screen.queryByText("Open external link")).toBeNull();
  });
});

// Streamdown parses each block on its own, so a reference definition never
// reaches the link it names in a rendered reply; the plugin is checked alone.
describe("remarkAppLinks", () => {
  it("rewrites the URL of a reference definition", () => {
    const definition = { type: "definition", url: "library/agents/1" };
    remarkAppLinks()({ type: "root", children: [definition] });

    expect(definition.url).toBe("/library/agents/1");
  });
});

describe("toAppPath", () => {
  it.each([
    ["library/agents/1", "/library/agents/1"],
    ["library", "/library"],
    ["library?tab=1", "/library?tab=1"],
    ["marketplace#top", "/marketplace#top"],
    ["/library/agents/1", "/library/agents/1"],
    ["//evil.com/library", "//evil.com/library"],
    ["\\\\evil.com", "\\\\evil.com"],
    ["./library", "./library"],
    ["#library", "#library"],
    ["library:x", "library:x"],
    ["streamdown:incomplete-link", "streamdown:incomplete-link"],
    ["libraryx/y", "libraryx/y"],
    ["Library/x", "Library/x"],
    ["example.com/library", "example.com/library"],
  ])("maps %s to %s", (url, expected) => {
    expect(toAppPath(url)).toBe(expected);
  });
});
