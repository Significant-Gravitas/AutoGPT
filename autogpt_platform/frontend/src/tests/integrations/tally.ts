import { act } from "@testing-library/react";
import { vi } from "vitest";

const TALLY_EMBED_URL = "https://tally.so/widgets/embed.js";

// happy-dom fails external scripts as soon as they are added, so the stub keeps
// Tally's script out of the document and lets the test settle it.
export function stubTallyEmbed() {
  const scripts: HTMLScriptElement[] = [];
  const unsettled = new Set<HTMLScriptElement>();
  const openPopup = vi.fn();
  const appendChild = document.head.appendChild.bind(document.head);
  const spy = vi
    .spyOn(document.head, "appendChild")
    .mockImplementation(function <T extends Node>(node: T): T {
      if (node instanceof HTMLScriptElement && node.src === TALLY_EMBED_URL) {
        scripts.push(node);
        unsettled.add(node);
        return node;
      }
      return appendChild(node);
    });

  async function settle(script: HTMLScriptElement, event: "load" | "error") {
    unsettled.delete(script);
    if (event === "load") window.Tally = { openPopup };
    await act(async () => {
      script.dispatchEvent(new Event(event));
    });
  }

  function lastScript() {
    const script = scripts.at(-1);
    if (!script) throw new Error("Tally's script was never requested");
    return script;
  }

  return {
    scripts,
    openPopup,
    finishLoading: () => settle(lastScript(), "load"),
    failLoading: () => settle(lastScript(), "error"),
    popupOptions: (call = 0) => openPopup.mock.calls[call][1],
    async restore() {
      for (const script of [...unsettled]) await settle(script, "error");
      spy.mockRestore();
      delete window.Tally;
    },
  };
}
