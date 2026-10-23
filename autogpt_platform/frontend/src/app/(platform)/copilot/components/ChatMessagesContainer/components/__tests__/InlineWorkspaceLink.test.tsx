import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
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

describe("an inline workspace link in a chat message", () => {
  let openSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    useCopilotUIStore.getState().closeArtifactPanel();
    openSpy = vi.spyOn(window, "open").mockReturnValue(null);
  });

  afterEach(() => {
    openSpy.mockRestore();
    cleanup();
  });

  it("opens the file in the artifact panel instead of downloading it", async () => {
    renderText(
      `Here is your doc: [Weekly research](workspace://${FILE_ID}#text/markdown).`,
    );

    fireEvent.click(await screen.findByText("Weekly research"));

    const { activeArtifact } = useCopilotUIStore.getState().artifactPanel;
    expect(activeArtifact?.id).toBe(FILE_ID);
    // Streamdown opens a link after an await, so give a leaked click its turn.
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(openSpy).not.toHaveBeenCalled();
  });

  it("keeps the download in the read-only share view and on a modified click", async () => {
    renderText(`[Weekly research](workspace://${FILE_ID}#text/markdown)`, true);
    fireEvent.click(await screen.findByText("Weekly research"));
    await waitFor(() => expect(openSpy).toHaveBeenCalledTimes(1));
    cleanup();
    renderText(`[Weekly research](workspace://${FILE_ID}#text/markdown)`);
    fireEvent.click(await screen.findByText("Weekly research"), {
      metaKey: true,
    });

    await waitFor(() => expect(openSpy).toHaveBeenCalledTimes(2));
    expect(
      useCopilotUIStore.getState().artifactPanel.activeArtifact,
    ).toBeNull();
  });
});
