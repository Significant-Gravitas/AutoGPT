import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
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

function Composer({ onSend }: { onSend: (message: string) => void }) {
  const [value, setValue] = useState("");
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
});
