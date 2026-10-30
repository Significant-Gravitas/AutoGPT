import { cleanup, render, waitFor } from "@testing-library/react";
import type { UIMessage } from "ai";
import { afterEach, describe, expect, it } from "vitest";
import { Message, MessageResponse } from "../Message";

const BOTH_BUBBLES = ["user", "assistant"] as const;

async function renderBubble(from: UIMessage["role"], markdown: string) {
  const { container } = render(
    <Message from={from}>
      <MessageResponse>{markdown}</MessageResponse>
    </Message>,
  );
  await waitFor(() => expect(container.textContent).not.toBe(""));
  return container;
}

describe("MessageResponse with text that looks like HTML", () => {
  afterEach(() => {
    cleanup();
  });

  it.each(BOTH_BUBBLES)(
    "shows placeholders as text in the %s's bubble",
    async (from) => {
      const container = await renderBubble(
        from,
        "Hi <first name>, welcome to <company>!\n\nThanks,\n\n<signature>",
      );

      expect(
        Array.from(container.querySelectorAll("p"), (p) => p.textContent),
      ).toEqual([
        "Hi <first name>, welcome to <company>!",
        "Thanks,",
        "<signature>",
      ]);
    },
  );

  it("shows the tags the user typed as text", async () => {
    const container = await renderBubble("user", "Keep <b>this</b> as typed");

    expect(container.querySelector("b")).toBeNull();
    expect(container.textContent).toBe("Keep <b>this</b> as typed");
  });

  it("renders the HTML Otto formats with", async () => {
    const container = await renderBubble(
      "assistant",
      "Keep <b>this</b> bold\n\n| Step | Notes |\n|---|---|\n| 1 | Sign in<br>Confirm |\n\n<details><summary>More</summary>\n\nHidden\n\n</details>",
    );

    expect(container.querySelector("p b")?.textContent).toBe("this");
    expect(container.querySelector("td br")).not.toBeNull();
    expect(container.querySelector("details summary")?.textContent).toBe(
      "More",
    );
  });

  it.each(BOTH_BUBBLES)(
    "still drops image handlers and script links in the %s's bubble",
    async (from) => {
      const container = await renderBubble(
        from,
        'An <img src=x onerror="alert(1)"> and <a href="javascript:alert(1)">this</a> or [that](javascript:alert(1))',
      );

      expect(container.querySelector("img")).toBeNull();
      expect(container.querySelector("[onerror]")).toBeNull();
      expect(container.querySelector('[data-streamdown="link"]')).toBeNull();
    },
  );

  it.each(BOTH_BUBBLES)(
    "leaves code and autolinks in the %s's bubble as they were",
    async (from) => {
      const container = await renderBubble(
        from,
        "Use `<first name>` or <https://example.com>\n\n```html\n<b>x</b>\n```",
      );

      expect(container.querySelector("p code")?.textContent).toBe(
        "<first name>",
      );
      expect(
        container.querySelector('[data-streamdown="link"]')?.textContent,
      ).toBe("https://example.com");
      await waitFor(() =>
        expect(
          container.querySelector('[data-streamdown="code-block-body"]')
            ?.textContent,
        ).toContain("<b>x</b>"),
      );
    },
  );

  it.each([
    ["user", "routines"],
    ["assistant", "thinking"],
  ] as const)(
    "keeps a server-only tag hidden in the %s's bubble: <%s>",
    async (from, tag) => {
      const container = await renderBubble(
        from,
        `<${tag}>\nnote\n</${tag}>\n\nAnswer`,
      );

      expect(container.textContent).not.toContain(`<${tag}>`);
      expect(container.textContent).toContain("Answer");
    },
  );
});
