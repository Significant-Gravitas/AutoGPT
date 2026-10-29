import { afterEach, describe, expect, it, vi } from "vitest";
import { getSpotlightBox } from "../helpers";

// jsdom lays nothing out: each element gets the box it is given here.
function laidOut(element: HTMLElement, rect: DOMRectInit) {
  vi.spyOn(element, "getBoundingClientRect").mockReturnValue(
    DOMRect.fromRect(rect),
  );
  return element;
}

function sidebarItem(collapsible: "icon" | "offcanvas") {
  const sidebar = document.createElement("div");
  sidebar.setAttribute("data-collapsible", collapsible);
  const item = document.createElement("li");
  const button = document.createElement("button");
  button.setAttribute("data-sidebar", "menu-button");
  item.append(button);
  sidebar.append(item);
  document.body.append(sidebar);
  return { item, button };
}

afterEach(() => {
  document.body.innerHTML = "";
  vi.restoreAllMocks();
});

describe("getSpotlightBox", () => {
  it("rings the icon evenly in the collapsed rail, not the rail-wide item", () => {
    const { item, button } = sidebarItem("icon");
    laidOut(item, { x: 0, y: 180, width: 47, height: 32 });
    laidOut(button, { x: 8, y: 180, width: 32, height: 32 });

    const box = getSpotlightBox(item, "team");

    expect(box).toEqual({ top: 176, left: 4, width: 40, height: 40 });
    const gapLeft = 8 - box.left;
    const gapRight = box.left + box.width - (8 + 32);
    expect(gapLeft).toBe(gapRight);
  });

  it("rings the whole item in the open sidebar", () => {
    const { item, button } = sidebarItem("offcanvas");
    laidOut(item, { x: 8, y: 180, width: 240, height: 32 });
    laidOut(button, { x: 8, y: 180, width: 240, height: 32 });

    expect(getSpotlightBox(item, "team")).toEqual({
      top: 174,
      left: 6,
      width: 248,
      height: 44,
    });
  });
});
