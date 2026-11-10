import { cleanup, render, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { MessageResponse } from "../Message";
import { STREAMING_CLASS } from "../stream-reveal";

afterEach(cleanup);

function wrapperOf(container: HTMLElement) {
  return container.firstElementChild as HTMLElement;
}

function carets(container: HTMLElement) {
  return Array.from(container.querySelectorAll("[data-stream-caret]"));
}

// The markup with the reveal spans taken out, to compare with a static render.
function withoutReveal(container: HTMLElement) {
  const clone = wrapperOf(container).cloneNode(true) as HTMLElement;
  clone.querySelectorAll("[data-stream-caret]").forEach((el) => el.remove());
  clone
    .querySelectorAll("[data-stream-fade]")
    .forEach((el) => el.replaceWith(...Array.from(el.childNodes)));
  clone.querySelectorAll(".streaming-caret-path").forEach((el) => {
    el.classList.remove("streaming-caret-path");
    if (el.classList.length === 0) el.removeAttribute("class");
  });
  clone.normalize();
  return clone.innerHTML;
}

describe("MessageResponse while the reply streams in", () => {
  it("puts one caret at the end of the last block's text", async () => {
    const { container } = render(
      <MessageResponse isStreaming>
        {"First paragraph.\n\nSecond one is still"}
      </MessageResponse>,
    );
    await waitFor(() => expect(container.textContent).toContain("still"));

    const wrapper = wrapperOf(container);
    expect(wrapper.className).toContain(STREAMING_CLASS);
    const last = wrapper.lastElementChild!;
    expect(last.tagName).toBe("P");
    expect(last.querySelectorAll("[data-stream-caret]")).toHaveLength(1);
    expect(last.lastElementChild?.hasAttribute("data-stream-caret")).toBe(true);
    expect(last.lastElementChild?.getAttribute("aria-hidden")).toBe("true");
    expect(last.lastElementChild?.textContent).toBe("");
  });

  it("keeps the words already shown and adds spans for the new ones", async () => {
    const { container, rerender } = render(
      <MessageResponse isStreaming>{"Hello there "}</MessageResponse>,
    );
    await waitFor(() => expect(container.textContent).toContain("there"));
    const before = Array.from(container.querySelectorAll("[data-stream-fade]"));
    expect(before.map((el) => el.textContent)).toEqual(["Hello ", "there"]);

    rerender(
      <MessageResponse isStreaming>
        {"Hello there general Kenobi "}
      </MessageResponse>,
    );
    await waitFor(() => expect(container.textContent).toContain("Kenobi"));
    const after = Array.from(container.querySelectorAll("[data-stream-fade]"));
    expect(after.length).toBeGreaterThan(before.length);
    // The same nodes, so their fade does not play again.
    expect(after[0]).toBe(before[0]);
    expect(carets(container)).toHaveLength(1);
  });

  it("removes the caret when the stream ends", async () => {
    const text = "Done now.\n\n- one\n- two";
    const { container, rerender } = render(
      <MessageResponse isStreaming>{text}</MessageResponse>,
    );
    // Each block carries one; STREAMING_CLASS shows only the last block's.
    await waitFor(() => expect(carets(container)).toHaveLength(2));
    expect(
      wrapperOf(container).lastElementChild?.querySelectorAll(
        "[data-stream-caret]",
      ),
    ).toHaveLength(1);

    rerender(<MessageResponse isStreaming={false}>{text}</MessageResponse>);
    await waitFor(() => expect(carets(container)).toHaveLength(0));
    expect(container.querySelectorAll(".streaming-caret-path")).toHaveLength(0);
    expect(wrapperOf(container).className).not.toContain(STREAMING_CLASS);
    const settled = render(<MessageResponse>{text}</MessageResponse>);
    await waitFor(() =>
      expect(container.textContent).toBe(settled.container.textContent),
    );
    expect(container.querySelectorAll("li")).toHaveLength(2);
  });

  it("shows nothing extra on a reply that is not streaming", async () => {
    const { container } = render(
      <MessageResponse>{"Settled reply"}</MessageResponse>,
    );
    await waitFor(() => expect(container.textContent).toContain("Settled"));
    expect(carets(container)).toHaveLength(0);
    expect(container.querySelectorAll("[data-stream-fade]")).toHaveLength(0);
  });

  it("splits Japanese and Chinese text into several fading words", async () => {
    const { container } = render(
      <MessageResponse isStreaming>
        {"これは日本語の文章です。\n\n我们今天去公园散步吧。"}
      </MessageResponse>,
    );
    await waitFor(() => expect(container.textContent).toContain("公园"));
    const [ja, zh] = Array.from(container.querySelectorAll("p"));
    expect(ja.querySelectorAll("[data-stream-fade]").length).toBeGreaterThan(3);
    expect(zh.querySelectorAll("[data-stream-fade]").length).toBeGreaterThan(3);
  });

  it("leaves an unfinished code block whole, with no caret after it", async () => {
    const { container } = render(
      <MessageResponse isStreaming>
        {"Here it is:\n\n```python\nprint('hi')\nx = 1"}
      </MessageResponse>,
    );
    await waitFor(() => expect(container.textContent).toContain("x = 1"));
    const code = container.querySelector("pre")?.parentElement ?? container;
    expect(code.querySelectorAll("[data-stream-fade]")).toHaveLength(0);
    expect(code.querySelectorAll("[data-stream-caret]")).toHaveLength(0);
  });

  it.each([
    ["an unfinished link", "Read [the docs](https://example.com/do"],
    ["unfinished bold", "This is **really important"],
    ["an unfinished table", "| Account | Status |\n| --- | --- |\n| Work | o"],
    ["a nested list", "1. first\n   - inner a\n   - inner b"],
    ["inline code", "Run `pnpm test` then"],
  ])("renders %s the same as without the reveal", async (_, markdown) => {
    const streaming = render(
      <MessageResponse isStreaming>{markdown}</MessageResponse>,
    );
    const settled = render(<MessageResponse>{markdown}</MessageResponse>);
    await waitFor(() =>
      expect(withoutReveal(streaming.container)).toBe(
        wrapperOf(settled.container).innerHTML,
      ),
    );
  });
});
