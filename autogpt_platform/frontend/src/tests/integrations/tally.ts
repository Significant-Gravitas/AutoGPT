import { act } from "@testing-library/react";
import { vi } from "vitest";

const TALLY_EMBED_URL = "https://tally.so/widgets/embed.js";

// Stands in for Tally's embed.js. happy-dom fails every external script as
// soon as it is added, so Tally's script is kept out of the document and the
// test decides when it loads or fails.
export function stubTallyEmbed() {
  const scripts: HTMLScriptElement[] = [];
  const openPopup = vi.fn();
  const appendChild = document.head.appendChild.bind(document.head);
  const spy = vi
    .spyOn(document.head, "appendChild")
    .mockImplementation(function <T extends Node>(node: T): T {
      if (node instanceof HTMLScriptElement && node.src === TALLY_EMBED_URL) {
        scripts.push(node);
        return node;
      }
      return appendChild(node);
    });

  async function settle(event: "load" | "error") {
    const script = scripts.at(-1);
    if (!script) throw new Error("Tally's script was never requested");
    if (event === "load") window.Tally = { openPopup };
    await act(async () => {
      script.dispatchEvent(new Event(event));
    });
  }

  return {
    scripts,
    openPopup,
    finishLoading: () => settle("load"),
    failLoading: () => settle("error"),
    popupOptions: (call = 0) => openPopup.mock.calls[call][1],
    restore() {
      spy.mockRestore();
      delete window.Tally;
    },
  };
}
