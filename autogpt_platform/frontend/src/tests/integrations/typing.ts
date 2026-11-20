import { fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect } from "vitest";

type TextField = HTMLInputElement | HTMLTextAreaElement;

/**
 * Types `text` into `field` at offset `at` (the end by default) and fails if
 * the field's text or caret was rewritten on the way. The keystrokes are
 * wrapped in composition events for fields that listen for them, but the
 * test environment can't run a real IME: what catches a broken field is the
 * text and caret check, since React writing its own value back moves the
 * caret to the end.
 *
 * A field that writes its own value over what was typed (reformatting it,
 * dropping an edit it can't parse yet, or getting it back one render late)
 * cancels every Japanese, Chinese or Korean composition and throws the caret
 * to the end. Use this in the test of any field that transforms or round-trips
 * its value.
 *
 * A number input has no caret and no IME, so only its text is checked, after
 * every keystroke.
 */
export async function expectTypingKept(
  field: TextField,
  text: string,
  at = field.value.length,
) {
  const expected = field.value.slice(0, at) + text + field.value.slice(at);
  if (field.type === "number") {
    typeIntoNumberInput(field, text, at);
    expect(field.value).toBe(expected);
    return;
  }
  const user = userEvent.setup();
  fireEvent.compositionStart(field);
  await user.type(field, escapeKeyDescriptors(text), {
    initialSelectionStart: at,
    initialSelectionEnd: at,
  });
  fireEvent.compositionEnd(field);
  expect(field.value).toBe(expected);
  expect(field.selectionStart).toBe(at + text.length);
}

// user-event writes a number input's text as `String(Number(text))`, which
// rounds integers past 2^53, so each keystroke is sent as a change event with
// the text typed so far, which the field must still hold afterwards.
function typeIntoNumberInput(field: TextField, text: string, at: number) {
  const before = field.value.slice(0, at);
  const after = field.value.slice(at);
  for (let end = 1; end <= text.length; end++) {
    const typed = before + text.slice(0, end) + after;
    fireEvent.change(field, { target: { value: typed } });
    expect(field.value, `after typing "${typed}"`).toBe(
      numberInputValue(typed),
    );
  }
}

// What a number input holds for `text`: text that isn't a number yet, such
// as a lone "-", reads as "".
function numberInputValue(text: string) {
  const input = document.createElement("input");
  input.type = "number";
  input.value = text;
  return input.value;
}

// user-event reads `{Enter}` and `[KeyA]` as key names; doubled, they type
// the bracket itself.
function escapeKeyDescriptors(text: string) {
  return text.replace(/[{[]/g, "$&$&");
}
