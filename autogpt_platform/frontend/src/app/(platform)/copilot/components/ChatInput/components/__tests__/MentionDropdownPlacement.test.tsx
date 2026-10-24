import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import type { MutableRefObject } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { MentionOption } from "../../useChatMentions";
import { MentionDropdown } from "../MentionDropdown";

const FILE_ITEM: MentionOption = {
  kind: "file",
  file: {
    id: "file-1",
    name: "alpha.txt",
    path: "/workspace/alpha.txt",
    mime_type: "text/plain",
    size_bytes: 10,
    origin: "uploaded",
    created_at: "2026-01-01T00:00:00Z",
  },
};

describe("MentionDropdown placement", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("opens above the composer when there is room above", async () => {
    stubLayout({ viewportHeight: 900, composer: { top: 760, height: 100 } });
    renderPlacedDropdown();
    const box = await waitForPlacedList();
    expect(box.bottom).toBeLessThanOrEqual(760 - 8);
    expect(box.top).toBeGreaterThanOrEqual(8);
  });

  it("opens below the composer when there is no room above", async () => {
    stubLayout({ viewportHeight: 900, composer: { top: 40, height: 100 } });
    renderPlacedDropdown();
    const box = await waitForPlacedList();
    expect(box.top).toBeGreaterThanOrEqual(140 + 8);
    expect(box.bottom).toBeLessThanOrEqual(900 - 8);
  });

  it("caps the list to the roomier side when neither side fits it", async () => {
    stubLayout({ viewportHeight: 420, composer: { top: 150, height: 100 } });
    renderPlacedDropdown();
    const box = await waitForPlacedList();
    expect(box.top).toBeGreaterThanOrEqual(250 + 8);
    expect(box.bottom).toBeLessThanOrEqual(420 - 8);
    expect(box.height).toBe(420 - 250 - 8 - 8);
  });

  it("caps the list to the composer's width on a narrow screen", async () => {
    stubLayout({
      viewportHeight: 568,
      viewportWidth: 320,
      composer: { top: 440, height: 100, left: 16, width: 288 },
    });
    renderPlacedDropdown();
    const box = await waitForPlacedList();
    expect(box.left).toBeGreaterThanOrEqual(16);
    expect(box.right).toBeLessThanOrEqual(16 + 288);
  });

  it("hides the list when its scrolling container hides the composer", async () => {
    stubLayout({
      viewportHeight: 900,
      composer: { top: -300, height: 100 },
      scroller: { top: 0, height: 600 },
    });
    renderPlacedDropdown({ inScroller: true });
    const list = screen.getByRole("listbox", { name: "Mention suggestions" });
    await waitFor(() => expect(list.style.visibility).toBe("hidden"));
  });

  it("keeps the list visible while the composer is inside its scrolling container", async () => {
    stubLayout({
      viewportHeight: 900,
      composer: { top: 300, height: 100 },
      scroller: { top: 0, height: 600 },
    });
    renderPlacedDropdown({ inScroller: true });
    await waitForPlacedList();
    const list = screen.getByRole("listbox", { name: "Mention suggestions" });
    expect(list.style.visibility).not.toBe("hidden");
  });
});

const LIST_CONTENT_HEIGHT = 300;

interface Box {
  top: number;
  height: number;
  left?: number;
  width?: number;
}

function stubLayout({
  viewportHeight,
  viewportWidth = 1440,
  composer,
  scroller,
}: {
  viewportHeight: number;
  viewportWidth?: number;
  composer: Box;
  scroller?: Box;
}) {
  function boxOf(element: Element): Box | null {
    if (element === document.documentElement) {
      return { top: 0, left: 0, height: viewportHeight, width: viewportWidth };
    }
    if (element.matches("[data-testid=composer]")) {
      return { left: 320, width: 800, ...composer };
    }
    if (element.matches("[data-testid=composer] > span[aria-hidden]")) {
      return { left: 320, width: 800, ...composer };
    }
    if (scroller && element.matches("[data-testid=scroller]")) {
      return { left: 0, width: viewportWidth, ...scroller };
    }
    if (element.matches("[role=listbox]")) {
      // Mirrors the list's max-h/max-w classes: 20rem, or less once the
      // hook writes the room it found into the CSS variables.
      const list = element as HTMLElement;
      const [x, y] = translateOf(list);
      return {
        top: y,
        left: x,
        height: Math.min(LIST_CONTENT_HEIGHT, cssCap(list, "height")),
        width: Math.min(320, cssCap(list, "width")),
      };
    }
    return null;
  }
  const original = Element.prototype.getBoundingClientRect;
  vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(
    function (this: Element) {
      const box = boxOf(this);
      if (!box) return original.call(this);
      return new DOMRect(box.left ?? 0, box.top, box.width ?? 0, box.height);
    },
  );
  function stubSize(
    prop: "offsetWidth" | "offsetHeight" | "clientWidth" | "clientHeight",
    pick: (box: Box) => number,
  ) {
    const descriptor = Object.getOwnPropertyDescriptor(
      HTMLElement.prototype,
      prop,
    ) ??
      Object.getOwnPropertyDescriptor(Element.prototype, prop) ?? {
        get: () => 0,
      };
    vi.spyOn(HTMLElement.prototype, prop, "get").mockImplementation(function (
      this: HTMLElement,
    ) {
      const box = boxOf(this);
      return box ? pick(box) : (descriptor.get?.call(this) ?? 0);
    });
  }
  stubSize("offsetWidth", (box) => box.width ?? 0);
  stubSize("offsetHeight", (box) => box.height);
  stubSize("clientWidth", (box) => box.width ?? 0);
  stubSize("clientHeight", (box) => box.height);
  vi.spyOn(HTMLElement.prototype, "scrollHeight", "get").mockImplementation(
    function (this: HTMLElement) {
      return this.matches("[role=listbox]") ? LIST_CONTENT_HEIGHT : 0;
    },
  );
}

function cssCap(list: HTMLElement, axis: "height" | "width") {
  const name =
    axis === "height" ? "--mention-available-height" : "--mention-anchor-width";
  return parseFloat(list.style.getPropertyValue(name)) || Infinity;
}

function translateOf(element: HTMLElement): [number, number] {
  const match = /translate\((-?[\d.]+)px, (-?[\d.]+)px\)/.exec(
    element.style.transform,
  );
  return match ? [Number(match[1]), Number(match[2])] : [0, 0];
}

function renderPlacedDropdown({ inScroller = false } = {}) {
  const composer = (
    <form data-testid="composer" className="relative">
      <MentionDropdown
        options={[FILE_ITEM]}
        showFiles
        hasIntegrations={false}
        isLoading={false}
        isError={false}
        highlightedIndex={0}
        highlightedRef={
          { current: null } as MutableRefObject<HTMLButtonElement | null>
        }
        onSelect={vi.fn()}
        onHighlight={vi.fn()}
      />
    </form>
  );
  render(
    inScroller ? (
      <div data-testid="scroller" style={{ overflowY: "auto" }}>
        {composer}
      </div>
    ) : (
      composer
    ),
  );
}

async function waitForPlacedList() {
  const list = screen.getByRole("listbox", { name: "Mention suggestions" });
  await waitFor(() =>
    expect(list.style.transform).toMatch(/translate\((?!0px, 0px\))/),
  );
  return list.getBoundingClientRect();
}
