import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { useChatInput } from "../../ChatInput/useChatInput";
import { CredentialMentionEditor } from "../CredentialMentionEditor";

const MESSAGE = "Check [Work Gmail](credential://google/work-credential-id)";

interface Props {
  onSend: (message: string) => void;
  onEnqueue?: (message: string) => Promise<void>;
  isStreaming?: boolean;
}

function Composer({ onSend, onEnqueue, isStreaming }: Props) {
  const input = useChatInput({ onSend, onEnqueue, isStreaming });
  return (
    <form onSubmit={input.handleSubmit}>
      <CredentialMentionEditor
        id="chat-input"
        value={input.value}
        onChange={input.setValue}
        onKeyDown={() => {}}
      />
      <button type="submit">Send</button>
    </form>
  );
}

afterEach(() => vi.restoreAllMocks());

describe("mobile composer focus", () => {
  it.each([false, true])(
    "keeps the keyboard dismissed after submitting a badge (streaming: %s)",
    async (isStreaming) => {
      vi.spyOn(window, "matchMedia").mockReturnValue({
        matches: true,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
      } as unknown as MediaQueryList);
      const onSend = vi.fn();
      const onEnqueue = vi.fn().mockResolvedValue(undefined);
      render(
        <Composer
          onSend={onSend}
          onEnqueue={onEnqueue}
          isStreaming={isStreaming}
        />,
      );
      const textarea = screen.getByRole("textbox");
      expect(document.activeElement).not.toBe(textarea);
      textarea.focus();
      fireEvent.change(textarea, { target: { value: MESSAGE } });
      const editor = screen.getByRole("textbox");
      expect(editor.tagName).toBe("DIV");
      expect(document.activeElement).toBe(editor);

      fireEvent.submit(
        screen.getByRole("button", { name: "Send" }).closest("form")!,
      );
      await waitFor(() =>
        expect(isStreaming ? onEnqueue : onSend).toHaveBeenCalledWith(MESSAGE),
      );
      const emptyInput = screen.getByRole<HTMLTextAreaElement>("textbox");
      expect(emptyInput.tagName).toBe("TEXTAREA");
      expect(emptyInput.value).toBe("");
      expect(document.activeElement).not.toBe(emptyInput);
    },
  );

  it("keeps desktop focus after submitting a badge", async () => {
    const user = userEvent.setup();
    vi.spyOn(window, "matchMedia").mockReturnValue({
      matches: false,
    } as MediaQueryList);
    const onSend = vi.fn();
    render(<Composer onSend={onSend} />);
    const textarea = screen.getByRole("textbox");
    expect(document.activeElement).toBe(textarea);
    fireEvent.change(textarea, { target: { value: MESSAGE } });
    await user.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(onSend).toHaveBeenCalledWith(MESSAGE));
    expect(document.activeElement).toBe(screen.getByRole("textbox"));
  });
});
