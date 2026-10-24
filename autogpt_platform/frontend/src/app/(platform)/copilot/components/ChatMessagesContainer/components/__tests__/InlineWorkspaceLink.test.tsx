import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { useCopilotUIStore } from "../../../../store";
import { MessagePartRenderer } from "../MessagePartRenderer";

type Part = UIMessage<unknown, UIDataTypes, UITools>["parts"][number];

// The card registers itself for auto-open, which would open the panel without
// a click; only the inline link is under test here.
vi.mock("../../../ArtifactCard/ArtifactCard", () => ({
  ArtifactCard: () => null,
}));

const FILE_ID = "0b3f1c2a-1111-4a2b-9c3d-123456789abc";
const DOC = `workspace://${FILE_ID}#text/markdown`;

function renderText(text: string, readOnly = false) {
  const part = { type: "text", text } as Part;
  render(
    <MessagePartRenderer
      part={part}
      messageID="m1"
      partIndex={0}
      readOnly={readOnly}
    />,
  );
}

// The element a link's whole label reads from, whatever formatting is inside it.
function isLinkReading(label: string) {
  return (_: string, element: Element | null) =>
    element?.textContent === label &&
    (element.tagName === "A" || element.tagName === "BUTTON");
}

function openedFileID() {
  return useCopilotUIStore.getState().artifactPanel.activeArtifact?.id;
}

describe("an inline workspace link in a chat message", () => {
  beforeEach(() => {
    useCopilotUIStore.getState().closeArtifactPanel();
  });

  afterEach(() => {
    cleanup();
  });

  it("opens the file in the artifact panel instead of downloading it", async () => {
    renderText(`Here is your doc: [Weekly research](${DOC}).`);

    const link = await screen.findByRole("link", { name: "Weekly research" });
    const downloaded = fireEvent.click(link);

    expect(openedFileID()).toBe(FILE_ID);
    expect(downloaded).toBe(false);
    expect(link.parentElement?.textContent).toBe(
      "Here is your doc: Weekly research.",
    );
  });

  it("leaves an external link with the same label to its own destination", async () => {
    renderText(`[Report](https://example.com/report) and [Report](${DOC})`);

    const [external, file] = await screen.findAllByText("Report");

    fireEvent.click(external);
    expect(await screen.findByText("Open external link")).toBeDefined();
    expect(openedFileID()).toBeUndefined();

    fireEvent.click(file);
    expect(openedFileID()).toBe(FILE_ID);
  });

  it("opens a file whose link label has formatting", async () => {
    renderText(`[**Weekly** \`research\`](${DOC})`);

    fireEvent.click(await screen.findByText(isLinkReading("Weekly research")));

    expect(openedFileID()).toBe(FILE_ID);
  });

  it("reads a link label's markdown escapes as the link text does", async () => {
    renderText(`[my\\_report.md](${DOC})`);

    expect(
      await screen.findByRole("link", { name: "my_report.md" }),
    ).toBeDefined();
  });

  it("keeps a download link to a file the message also shows as an image", async () => {
    const image = `workspace://${FILE_ID}#image/png`;
    renderText(`![Chart](${image}) Full size: [Chart file](${image})`);

    const link = await screen.findByRole("link", { name: "Chart file" });

    expect(link.getAttribute("href")).toContain(`/files/${FILE_ID}/download`);
  });

  it("keeps the download in the read-only share view and on a modified click", async () => {
    renderText(`[Weekly research](${DOC})`, true);
    const shared = await screen.findByRole("link", { name: "Weekly research" });
    expect(fireEvent.click(shared)).toBe(true);
    cleanup();
    renderText(`[Weekly research](${DOC})`);
    const link = await screen.findByRole("link", { name: "Weekly research" });

    expect(fireEvent.click(link, { metaKey: true })).toBe(true);
    expect(openedFileID()).toBeUndefined();
    expect(link.getAttribute("href")).toContain(`/files/${FILE_ID}/download`);
  });
});
