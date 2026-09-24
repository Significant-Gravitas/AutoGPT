import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { CredentialMentionEditor } from "../CredentialMentionEditor";
import { useChatMentions } from "../../ChatInput/useChatMentions";
import { connectedIntegrationsFromCredentials } from "../../ChatInput/helpers";

const accounts = connectedIntegrationsFromCredentials([
  {
    id: "work-credential-id",
    provider: "google",
    type: "oauth2",
    title: "Work Gmail",
    scopes: null,
    username: null,
  },
]);

const WORK = "[Work Gmail](credential://google/work-credential-id)";
const PERSONAL = "[Personal](credential://google/personal-credential-id)";

function Composer({
  onSend,
  initialValue = "",
}: {
  onSend: (message: string) => void;
  initialValue?: string;
}) {
  const [value, setValue] = useState(initialValue);
  const mentions = useChatMentions({
    enabled: true,
    value,
    setValue,
    integrations: accounts,
    includeWorkspaceFiles: false,
    addWorkspaceFile: vi.fn(),
    addWorkspaceFolder: vi.fn(),
  });
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        onSend(String(new FormData(event.currentTarget).get("message")));
      }}
    >
      <CredentialMentionEditor
        value={value}
        onInputReady={mentions.bindInput}
        onChange={(next, input) => {
          setValue(next);
          mentions.detect(input);
        }}
        onKeyDown={(event) => {
          mentions.onKeyDown(event);
        }}
      />
      <button type="submit">Send</button>
    </form>
  );
}

function placeCaret(node: Node, offset: number) {
  const range = document.createRange();
  range.setStart(node, offset);
  range.collapse(true);
  const selection = window.getSelection();
  selection?.removeAllRanges();
  selection?.addRange(range);
}

function badgesIn(editor: HTMLElement) {
  return Array.from(editor.querySelectorAll("[data-credential-mention]"));
}

describe("account badge composer", () => {
  it("inserts a visible badge and submits the corresponding credential ID", async () => {
    const onSend = vi.fn();
    render(<Composer onSend={onSend} />);
    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: "Check @Work", selectionStart: 11 },
    });
    fireEvent.keyDown(screen.getByRole("textbox"), { key: "Enter" });
    const editor = screen.getByRole("textbox");
    await waitFor(() => expect(editor.textContent).toBe("Check Work Gmail "));
    expect(editor.textContent).not.toContain("credential-id");
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(onSend).toHaveBeenCalledWith(
      "Check [Work Gmail](credential://google/work-credential-id) ",
    );
  });

  it("removes an account reference when its badge is deleted", async () => {
    const onSend = vi.fn();
    render(<Composer onSend={onSend} />);
    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: "Check @Work", selectionStart: 11 },
    });
    fireEvent.keyDown(screen.getByRole("textbox"), { key: "Enter" });
    const editor = screen.getByRole("textbox");
    await waitFor(() => expect(editor.textContent).toContain("Work Gmail"));
    editor.querySelector("[data-credential-mention]")?.remove();
    fireEvent.input(editor);
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(onSend).toHaveBeenCalledWith("Check  ");
  });

  it("keeps the caret at the deletion site when the last badge is removed mid-text", async () => {
    const onSend = vi.fn();
    const user = userEvent.setup();
    render(<Composer onSend={onSend} initialValue={`Prefix ${WORK} suffix`} />);
    const editor = screen.getByRole("textbox");
    expect(editor.textContent).toBe("Prefix Work Gmail suffix");
    editor.focus();
    badgesIn(editor)[0].remove();
    placeCaret(editor.firstChild!, "Prefix ".length);
    fireEvent.input(editor);

    const textarea = screen.getByRole<HTMLTextAreaElement>("textbox");
    expect(textarea.tagName).toBe("TEXTAREA");
    expect(textarea.value).toBe("Prefix  suffix");
    expect(document.activeElement).toBe(textarea);
    expect(textarea.selectionStart).toBe("Prefix ".length);
    expect(textarea.selectionEnd).toBe("Prefix ".length);

    await user.keyboard("X");
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(onSend).toHaveBeenCalledWith("Prefix X suffix");
  });

  it("keeps the caret after text typed over a selection spanning the last badge", () => {
    const onSend = vi.fn();
    render(<Composer onSend={onSend} initialValue={`Prefix ${WORK} suffix`} />);
    const editor = screen.getByRole("textbox");
    editor.focus();
    editor.replaceChildren(
      document.createTextNode("PreX"),
      document.createTextNode("fix"),
    );
    placeCaret(editor.firstChild!, "PreX".length);
    fireEvent.input(editor);

    const textarea = screen.getByRole<HTMLTextAreaElement>("textbox");
    expect(textarea.tagName).toBe("TEXTAREA");
    expect(textarea.value).toBe("PreXfix");
    expect(textarea.selectionStart).toBe("PreX".length);
    expect(textarea.selectionEnd).toBe("PreX".length);
  });

  it("restores the caret when typing a reference turns the textarea into a rich editor", () => {
    render(<Composer onSend={vi.fn()} initialValue="Check  now" />);
    const textarea = screen.getByRole<HTMLTextAreaElement>("textbox");
    textarea.focus();
    fireEvent.change(textarea, {
      target: {
        value: `Check ${WORK} now`,
        selectionStart: `Check ${WORK}`.length,
        selectionEnd: `Check ${WORK}`.length,
      },
    });
    const editor = screen.getByRole("textbox");
    expect(editor.tagName).toBe("DIV");
    expect(editor.textContent).toBe("Check Work Gmail now");
    expect(document.activeElement).toBe(editor);
    const range = window.getSelection()!.getRangeAt(0);
    expect(range.collapsed).toBe(true);
    expect(range.comparePoint(badgesIn(editor)[0], 0)).toBe(-1);
    expect(range.comparePoint(editor.lastChild!, 1)).toBe(1);
  });

  it("turns a reference pasted into a rich editor into a badge without exposing its ID", () => {
    const onSend = vi.fn();
    render(<Composer onSend={onSend} initialValue={`Check ${WORK} `} />);
    const editor = screen.getByRole("textbox");
    editor.focus();
    const trailing = editor.lastChild as Text;
    trailing.textContent += PERSONAL;
    placeCaret(trailing, trailing.length);
    fireEvent.input(editor);

    const badges = badgesIn(editor);
    expect(badges).toHaveLength(2);
    expect(editor.textContent).toBe("Check Work Gmail Personal");
    expect(editor.textContent).not.toContain("credential");
    const range = window.getSelection()!.getRangeAt(0);
    expect(range.collapsed).toBe(true);
    expect(editor.contains(range.startContainer)).toBe(true);
    expect(range.comparePoint(badges[1], 0)).toBe(-1);
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(onSend).toHaveBeenCalledWith(`Check ${WORK} ${PERSONAL}`);
  });

  it("defers turning a reference into a badge until IME composition ends", () => {
    render(<Composer onSend={vi.fn()} initialValue={`Check ${WORK} `} />);
    const editor = screen.getByRole("textbox");
    editor.focus();
    const trailing = editor.lastChild as Text;
    fireEvent.compositionStart(editor);
    trailing.textContent += PERSONAL;
    placeCaret(trailing, trailing.length);
    fireEvent.input(editor);
    expect(badgesIn(editor)).toHaveLength(1);
    expect(editor.lastChild).toBe(trailing);

    fireEvent.compositionEnd(editor);
    expect(badgesIn(editor)).toHaveLength(2);
    expect(editor.textContent).toBe("Check Work Gmail Personal");
  });

  it("submits one newline for a browser-inserted empty line", () => {
    const onSend = vi.fn();
    render(<Composer onSend={onSend} initialValue={`Check ${WORK}`} />);
    const editor = screen.getByRole("textbox");
    editor.focus();
    const line = document.createElement("div");
    line.appendChild(document.createElement("br"));
    editor.appendChild(line);
    placeCaret(line, 0);
    fireEvent.input(editor);
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(onSend).toHaveBeenCalledWith(`Check ${WORK}\n`);
  });
});
