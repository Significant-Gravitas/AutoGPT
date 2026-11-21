import { act, render, screen } from "@testing-library/react";
import { Component, type ReactNode, useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const capturePostHogEvent = vi.hoisted(() => vi.fn());
vi.mock("@/services/analytics/posthog-capture", () => ({
  capturePostHogEvent,
}));

const nativeRemoveChild = Node.prototype.removeChild;
const nativeInsertBefore = Node.prototype.insertBefore;

// A fresh module per test, so "once per page load" starts over.
beforeEach(async () => {
  vi.resetModules();
  capturePostHogEvent.mockClear();
  const { toleratePageTranslators } = await import("../page-translator-guard");
  toleratePageTranslators();
});

afterEach(() => {
  Node.prototype.removeChild = nativeRemoveChild;
  Node.prototype.insertBefore = nativeInsertBefore;
  vi.restoreAllMocks();
});

describe("a page a translator has rewritten", () => {
  it("survives React removing a text node the translator replaced", () => {
    let hideNote: () => void = () => {};
    function Task() {
      const [showNote, setShowNote] = useState(true);
      hideNote = () => setShowNote(false);
      return (
        <p data-testid="task">
          {showNote ? "Draft the weekly report" : null}
          <b>Due Friday</b>
        </p>
      );
    }

    renderInErrorBoundary(<Task />);
    translateText(screen.getByTestId("task"), "Draft the weekly report");

    act(() => hideNote());

    expect(screen.queryByText("Something went wrong")).toBeNull();
    expect(screen.getByText("Due Friday")).toBeDefined();
  });

  it("survives React inserting before a text node the translator replaced", () => {
    let showBadge: () => void = () => {};
    function Task() {
      const [badge, setBadge] = useState(false);
      showBadge = () => setBadge(true);
      return (
        <p data-testid="task">
          {badge ? <span>New</span> : null}
          {"Draft the weekly report"}
          <b>Due Friday</b>
        </p>
      );
    }

    renderInErrorBoundary(<Task />);
    translateText(screen.getByTestId("task"), "Draft the weekly report");

    act(() => showBadge());

    expect(screen.queryByText("Something went wrong")).toBeNull();
    expect(screen.getByText("Due Friday")).toBeDefined();
  });
});

describe("toleratePageTranslators", () => {
  it("passes calls on the parent's own children through", () => {
    const parent = document.createElement("ul");
    const first = parent.appendChild(listItem("first"));
    const last = parent.appendChild(listItem("last"));
    const inserted = listItem("inserted");

    expect(parent.insertBefore(inserted, last)).toBe(inserted);
    expect(parent.removeChild(first)).toBe(first);
    parent.insertBefore(listItem("appended"), null);

    expect(Array.from(parent.children, (item) => item.textContent)).toEqual([
      "inserted",
      "last",
      "appended",
    ]);
  });

  it("skips calls on a node that has moved, and reports the first once without logging", async () => {
    const consoleSpies = [
      vi.spyOn(console, "error"),
      vi.spyOn(console, "warn"),
    ];
    const parent = document.createElement("p");
    parent.append("Due Friday");
    const moved = document.createTextNode("Draft the weekly report");

    expect(parent.removeChild(moved)).toBe(moved);
    const inserted = document.createElement("span");
    expect(parent.insertBefore(inserted, moved)).toBe(inserted);

    expect(parent.innerHTML).toBe("Due Friday");
    await vi.waitFor(() => expect(capturePostHogEvent).toHaveBeenCalled());
    expect(capturePostHogEvent).toHaveBeenCalledTimes(1);
    expect(capturePostHogEvent).toHaveBeenCalledWith("dom_call_skipped", {
      call: "removeChild",
      path: window.location.pathname,
    });
    for (const spy of consoleSpies) expect(spy).not.toHaveBeenCalled();
  });
});

function listItem(text: string) {
  const item = document.createElement("li");
  item.textContent = text;
  return item;
}

// Chrome's translator swaps each text node for nested <font> elements holding
// the translation; React keeps the original, now detached, node.
function translateText(container: HTMLElement, text: string) {
  const original = Array.from(container.childNodes).find(
    (node) => node.nodeType === Node.TEXT_NODE && node.textContent === text,
  );
  if (!original) throw new Error(`no text node "${text}"`);
  const outer = document.createElement("font");
  const inner = document.createElement("font");
  outer.style.verticalAlign = "inherit";
  inner.style.verticalAlign = "inherit";
  inner.textContent = "Redactar el informe semanal";
  outer.append(inner);
  original.replaceWith(outer);
}

function renderInErrorBoundary(children: ReactNode) {
  return render(<ErrorBoundary>{children}</ErrorBoundary>);
}

class ErrorBoundary extends Component<
  { children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? (
      <p>Something went wrong</p>
    ) : (
      this.props.children
    );
  }
}
