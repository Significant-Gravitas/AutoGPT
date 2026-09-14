import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { PromptInput, PromptInputTextarea } from "../prompt-input";

function renderComposer() {
  const onSubmit = vi.fn();
  render(
    <PromptInput onSubmit={onSubmit}>
      <PromptInputTextarea
        aria-label="Message"
        value="こんにちは"
        onChange={() => {}}
      />
    </PromptInput>,
  );
  return { textarea: screen.getByLabelText("Message"), onSubmit };
}

describe("PromptInputTextarea Enter handling", () => {
  it("submits the message on a plain Enter", () => {
    const { textarea, onSubmit } = renderComposer();

    fireEvent.keyDown(textarea, { key: "Enter" });

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit.mock.calls[0][0]).toBe("こんにちは");
  });

  it("does not submit while an IME is composing", () => {
    const { textarea, onSubmit } = renderComposer();

    fireEvent.keyDown(textarea, { key: "Enter", isComposing: true });
    fireEvent.keyDown(textarea, { key: "Enter", keyCode: 229 });

    expect(onSubmit).not.toHaveBeenCalled();
  });

  // This component used to track composition itself with
  // onCompositionStart/onCompositionEnd state. That layer is gone, so walk the
  // full browser sequence and prove the keydown flags alone are sufficient.
  it("submits only after the composition has ended", () => {
    const { textarea, onSubmit } = renderComposer();

    fireEvent.compositionStart(textarea, { data: "ねこ" });
    // Chrome and Firefox flag the confirming Enter itself.
    fireEvent.keyDown(textarea, { key: "Enter", isComposing: true });
    expect(onSubmit).not.toHaveBeenCalled();

    fireEvent.compositionEnd(textarea, { data: "猫" });
    // Safari fires its confirming Enter after compositionend, with
    // isComposing already false and only keyCode 229 left to go on.
    fireEvent.keyDown(textarea, { key: "Enter", keyCode: 229 });
    expect(onSubmit).not.toHaveBeenCalled();

    // The next Enter is the user asking to send.
    fireEvent.keyDown(textarea, { key: "Enter", keyCode: 13 });
    expect(onSubmit).toHaveBeenCalledTimes(1);
  });

  it("does not submit on Shift+Enter", () => {
    const { textarea, onSubmit } = renderComposer();

    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: true });

    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("lets a consumer onKeyDown claim the key first", () => {
    const onSubmit = vi.fn();
    render(
      <PromptInput onSubmit={onSubmit}>
        <PromptInputTextarea
          aria-label="Message"
          value="hi"
          onChange={() => {}}
          onKeyDown={(e) => e.preventDefault()}
        />
      </PromptInput>,
    );

    fireEvent.keyDown(screen.getByLabelText("Message"), { key: "Enter" });

    expect(onSubmit).not.toHaveBeenCalled();
  });
});
