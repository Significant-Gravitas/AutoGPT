import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderToString } from "react-dom/server";
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
    container.querySelectorAll<HTMLElement>('[data-streamdown="link"]'),
  );
}

function auxClick(link: HTMLElement, button: number) {
  return fireEvent(
    link,
    new MouseEvent("auxclick", { bubbles: true, cancelable: true, button }),
  );
}

// The ways of opening a link. A Ctrl-click on macOS is a right-click, which
// fires no click; this is the Ctrl-click of Windows and Linux.
const CLICKS: [string, (link: HTMLElement) => void][] = [
  ["a plain click", (link) => fireEvent.click(link)],
  ["a middle-click", (link) => auxClick(link, 1)],
  ["a Ctrl-click", (link) => fireEvent.click(link, { ctrlKey: true })],
  ["a Cmd-click", (link) => fireEvent.click(link, { metaKey: true })],
  ["a Shift-click", (link) => fireEvent.click(link, { shiftKey: true })],
  ["an Alt-click", (link) => fireEvent.click(link, { altKey: true })],
];

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

  it.each(CLICKS)(
    "leaves %s on a same-site link to the browser",
    async (_, clickLink) => {
      const container = await renderReply(
        `[open it](/library/agents/${AGENT_ID})`,
      );
      const link = screen.getByRole("link", { name: "open it" });

      expect(link.getAttribute("target")).toBe("_blank");
      expect(leavesToBrowser(link, clickLink)).toBe(true);
      expect(screen.queryByText("Open external link")).toBeNull();
      expect(window.open).not.toHaveBeenCalled();
      expect(linksOf(container)).toEqual([link]);
    },
  );

  it("draws a link to another site as a button without an href", async () => {
    const container = await renderReply("[docs](https://docs.agpt.co/guide)");
    const link = screen.getByRole("button", { name: "docs" });

    expect(linksOf(container)).toEqual([link]);
    expect(link.hasAttribute("href")).toBe(false);
    expect(screen.queryByRole("link", { name: "docs" })).toBeNull();
  });

  it.each(CLICKS)(
    "asks before %s leaves the site, then opens a new tab",
    async (_, clickLink) => {
      await renderReply("[docs](https://docs.agpt.co/guide)");
      const link = screen.getByRole("button", { name: "docs" });

      clickLink(link);
      expect(
        await screen.findByText("https://docs.agpt.co/guide"),
      ).toBeTruthy();
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
    },
  );

  it("treats an absolute link to this site as a same-site link", async () => {
    const url = `${window.location.origin}/library/agents/${AGENT_ID}`;
    await renderReply(`[open it](${url})`);

    expect(hrefOf(screen.getByRole("link", { name: "open it" }))).toBe(url);
  });

  it("never links a protocol-relative URL to the other site", async () => {
    const container = await renderReply("[x](//evil.example/path)");

    expect(screen.queryByRole("button", { name: "x" })).toBeNull();
    for (const link of linksOf(container)) {
      expect(hrefOf(link) ?? "").not.toContain("evil.example");
    }
  });

  it("asks when a link to another site is opened from the keyboard", async () => {
    await renderReply("[docs](https://docs.agpt.co/guide)");
    screen.getByRole("button", { name: "docs" }).focus();

    await userEvent.keyboard("{Enter}");

    expect(await screen.findByText("https://docs.agpt.co/guide")).toBeTruthy();
    expect(window.open).not.toHaveBeenCalled();
  });

  it("keeps the dialog when the caller passes its own link renderer", async () => {
    render(
      <MessageResponse components={{ a: () => <a href="#">custom</a> }}>
        {"[docs](https://docs.agpt.co/guide)"}
      </MessageResponse>,
    );

    fireEvent.click(await screen.findByRole("button", { name: "docs" }));
    expect(await screen.findByText("https://docs.agpt.co/guide")).toBeTruthy();
    expect(screen.queryByText("custom")).toBeNull();
  });

  it("opens nothing when the dialog is cancelled", async () => {
    await renderReply("[docs](https://docs.agpt.co/guide)");

    fireEvent.click(screen.getByRole("button", { name: "docs" }));
    fireEvent.click(await screen.findByRole("button", { name: "Cancel" }));

    await waitFor(() =>
      expect(screen.queryByText("Open external link")).toBeNull(),
    );
    expect(window.open).not.toHaveBeenCalled();
  });

  it("leaves a right-click on a link to another site to the context menu", async () => {
    await renderReply("[docs](https://docs.agpt.co/guide)");
    const link = screen.getByRole("button", { name: "docs" });

    fireEvent.contextMenu(link, { button: 2 });
    auxClick(link, 2);

    expect(screen.queryByText("Open external link")).toBeNull();
    expect(window.open).not.toHaveBeenCalled();
  });

  it("holds back autoscroll on a middle press on a link to another site", async () => {
    await renderReply("[docs](https://docs.agpt.co/guide)");
    const link = screen.getByRole("button", { name: "docs" });

    expect(fireEvent.mouseDown(link, { button: 1 })).toBe(false);
    expect(fireEvent.mouseDown(link, { button: 0 })).toBe(true);
  });

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

  it("keeps a link with a scheme it can't open inert", () => {
    render(<MessageLink href="javascript:alert(1)">run</MessageLink>);
    const link = screen.getByText("run");

    expect(link.hasAttribute("href")).toBe(false);
    expect(screen.queryByRole("button", { name: "run" })).toBeNull();
    fireEvent.click(link);
    expect(screen.queryByText("Open external link")).toBeNull();
  });

  it("asks before opening an email link", () => {
    render(<MessageLink href="mailto:help@example.com">write</MessageLink>);

    fireEvent.click(screen.getByRole("button", { name: "write" }));
    expect(screen.getByText("mailto:help@example.com")).toBeTruthy();
  });

  it("keeps the link's title on a link to another site", () => {
    render(
      <MessageLink href="https://docs.agpt.co" title="The guide">
        docs
      </MessageLink>,
    );

    expect(
      screen.getByRole("button", { name: "docs" }).getAttribute("title"),
    ).toBe("The guide");
  });

  it("draws a same-site path as an anchor on the server too", () => {
    const html = renderToString(
      <MessageLink href={`/library/agents/${AGENT_ID}`}>open it</MessageLink>,
    );

    expect(html).toContain(`href="/library/agents/${AGENT_ID}"`);
    expect(html).not.toContain("<button");
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
