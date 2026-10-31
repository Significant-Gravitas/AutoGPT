import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { remarkAppLinks } from "../app-links";
import { MessageResponse } from "../Message";

const AGENT_ID = "144760ff-de7d-4cc2-94ee-4bccaa741cf7";

async function renderReply(markdown: string) {
  const { container } = render(<MessageResponse>{markdown}</MessageResponse>);
  await waitFor(() => expect(container.textContent).not.toBe(""));
  return container;
}

// With link safety on, Streamdown renders every link as a button that opens
// same-site URLs at once and asks before leaving the site.
function linksOf(container: HTMLElement) {
  return Array.from(
    container.querySelectorAll<HTMLButtonElement>('[data-streamdown="link"]'),
  );
}

async function openedUrlOf(link: HTMLButtonElement) {
  fireEvent.click(link);
  await waitFor(() => expect(window.open).toHaveBeenCalledTimes(1));
  return vi.mocked(window.open).mock.calls[0][0];
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
    expect(await openedUrlOf(links[0])).toBe(`/library/agents/${AGENT_ID}`);
  });

  it("leaves an app path with a leading slash as it is", async () => {
    const container = await renderReply(
      `[open it](/library/agents/${AGENT_ID})`,
    );

    expect(await openedUrlOf(linksOf(container)[0])).toBe(
      `/library/agents/${AGENT_ID}`,
    );
  });

  it("leaves an absolute URL as it is", async () => {
    const container = await renderReply("[docs](https://docs.agpt.co/guide)");

    fireEvent.click(linksOf(container)[0]);
    expect(await screen.findByText("https://docs.agpt.co/guide")).toBeTruthy();
    expect(window.open).not.toHaveBeenCalled();
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

// Streamdown parses each block on its own, so a reference definition never
// reaches the link it names in a rendered reply; the plugin is checked alone.
describe("remarkAppLinks", () => {
  it("rewrites the URL of a reference definition", () => {
    const definition = { type: "definition", url: "library/agents/1" };
    remarkAppLinks()({ type: "root", children: [definition] });

    expect(definition.url).toBe("/library/agents/1");
  });
});
