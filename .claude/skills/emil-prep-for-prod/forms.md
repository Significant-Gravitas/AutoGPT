# Lane D — Forms

Forms are where users do real work and where launch-day money is lost: signup, checkout, invite, settings. Every check here is a rough edge someone hits in their first session. For anything deeper, the owning skill is `emil-forms-and-inputs`.

Sweep the forms in order of what they cost when broken: auth and checkout first, then creation flows, then settings.

Label association (A4), color-only errors (A7), 16px inputs (C3), and autofocus on touch (C13) live in their own lanes. Count each once.

## D1. Submission can't double-fire — BLOCKER

A submit button that stays enabled sends the request twice on a double-click or an impatient second tap: two charges, two invites, two projects. The button disables while the request is in flight and its label says what's happening.

**Hunt for:** submit handlers with no `isSubmitting` / `isPending` / `useFormStatus` state; `<button type="submit">` with no `disabled` prop; async `onClick` handlers on "Create", "Pay", "Send", "Invite", "Delete".

```jsx
<button type="submit" disabled={isSubmitting}>
  {isSubmitting ? "Saving…" : "Save changes"}
</button>
```

Keep the button's width stable while the label changes, or the layout twitches on every submit (lane E). The `finally` branch re-enables the button; a form that stays disabled forever after a failed request is the same bug from the other side.

## D2. Inputs live in a `<form>` — BLOCKER on auth and checkout, SHOULD FIX otherwise

Without a wrapping `<form>`, Enter does nothing, password managers can't find the fields, and the mobile keyboard's return key is dead. The user types their password, hits Enter, and the page sits there.

**Hunt for:** inputs with a submit `onClick` on a sibling button and no `<form onSubmit>` ancestor; `<form>` with no `onSubmit`; buttons inside a form with no `type` (a bare `<button>` is a submit button, so "Cancel" submits).

```jsx
<form onSubmit={handleSubmit}>
  <input type="email" name="email" />
  <button type="button" onClick={onCancel}>Cancel</button>
  <button type="submit">Continue</button>
</form>
```

Textareas submit on Cmd+Enter (Mac) / Ctrl+Enter (Windows), since Enter inserts a newline:

```jsx
<textarea onKeyDown={(e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") handleSubmit(e);
}} />
```

## D3. A failed submit says so, and keeps what was typed — BLOCKER

The happy path was tested. The unhappy path is a swallowed `catch`, a spinner that never stops, or a form that clears itself on error and makes the user retype everything.

**Hunt for:** `catch` blocks that only `console.error`; `try` with no `catch` around a `fetch`; no `finally` resetting the submitting state; form `reset()` called before the response is checked; no handling for a non-2xx response (`fetch` doesn't throw on a 500).

Fix: on failure, keep every field's value, re-enable the button, and show a message next to the form that says what happened and what to do ("We couldn't save your changes. Check your connection and try again."). Field-level server errors ("That email is already registered") render next to their field (D5).

## D4. Validation waits for the user to finish — SHOULD FIX

Flagging a field as invalid while the user is still typing their first attempt yells "invalid email" after one character. Validate on blur or on submit. Once a field *has* shown an error, switch to validating on change so the error clears the moment they fix it: reward early, punish late.

**Hunt for:** validation run inside `onChange` with no touched or submitted gate; schema resolvers configured with `mode: "onChange"` (react-hook-form: use `mode: "onBlur"` or `"onTouched"` with `reValidateMode: "onChange"`).

## D5. Errors sit next to their field and name the fix — SHOULD FIX

An error summary at the top of the form makes the user hunt. Each message renders directly under the field that caused it, wired with `aria-describedby`, and tells the user how to fix it.

**Hunt for:** a single `{error && <Alert>}` at the top of a multi-field form; messages like "Invalid input", "Error", "Required", "Validation failed".

| Instead of | Write |
| --- | --- |
| "Invalid input" | "Your email must include an @" |
| "Required" | "Enter your company name" |
| "Password error" | "Use at least 8 characters" |

After a failed submit, move focus to the first invalid field so the keyboard and screen reader user lands on the problem.

## D6. Inputs carry the right `type` and `autocomplete` — SHOULD FIX

The right `type` gets you the right mobile keyboard, browser validation, and autofill for free. The right `autocomplete` lets the browser and password manager fill the form in one tap, which on a phone is the difference between a signup and a bounce.

**Hunt for:** `type="text"` on email, phone, URL, and search fields; password fields with no `autocomplete`; `autocomplete="off"` on identity, address, or payment fields.

```html
<input type="email"    autocomplete="email" />
<input type="password" autocomplete="current-password" />  <!-- login -->
<input type="password" autocomplete="new-password" />      <!-- signup, change password -->
<input type="text"     autocomplete="one-time-code" inputmode="numeric" />
<input type="tel"      autocomplete="tel" />
```

Turn the noise **off** where it doesn't help: `spellcheck="false"` and `autocomplete="off"` on usernames, slugs, search fields, API keys, and codes; `data-1p-ignore` / `data-lpignore="true"` on fields where a password-manager overlay doesn't belong (a search box, a project name).

## D7. Destructive actions confirm — SHOULD FIX, BLOCKER when irreversible with no undo

Delete, remove, revoke, cancel subscription, leave team: a confirmation step before anything irreversible, or an undo window after it. A proper confirmation dialog, not `window.confirm()`. The confirm button names the outcome ("Delete project"), not "OK" or "Yes".

**Hunt for:** handlers named `delete*` / `remove*` / `revoke*` / `cancel*` wired straight to a button; `window.confirm(`; a destructive button styled the same as the primary and sitting next to it.

Adding a confirmation changes behavior, so when none exists apply it only if the project already has a dialog primitive and a pattern to follow. Otherwise it goes on the human list.

## D8. Buttons respond to a press — SHOULD FIX

A button with no pressed state feels unresponsive, and on touch (where hover doesn't exist) it gives no feedback at all until the action completes.

**Hunt for:** the shared `Button` component with `:hover` but no `:active`.

```css
.button { transition: transform 100ms ease-out; }
.button:active { transform: scale(0.97); }
```

Never below `0.95`; it reads as exaggerated.

## D9. No dead zones in checkbox and radio rows — SHOULD FIX

The label, the control, and the gap between them must all be clickable. A 16px checkbox with a label that doesn't toggle it is a tap target nobody can hit.

**Hunt for:** checkbox/radio `<input>` with a sibling `<span>` or `<p>` instead of a `<label>`; flex `gap` between an input and its `<label for>`.

```html
<label class="checkbox-row">
  <input type="checkbox" />
  <span>Remember me</span>
</label>
```

## D10. Input decorations sit inside the input — SHOULD FIX

A prefix icon rendered as a *sibling* of the input, inside a wrapper styled to look like the input, lies about the hit area: clicking the icon does nothing. Decorations are absolutely positioned over the real input, with padding making room. Non-interactive ones get `pointer-events: none`; interactive ones (a clear button) refocus the input after acting.

**Hunt for:** wrappers with `border` + `focus-within` styling containing an icon and a borderless input side by side.

```css
.input-wrapper { position: relative; }
.input-icon {
  position: absolute;
  left: 12px;
  top: 50%;
  transform: translateY(-50%);
  pointer-events: none;
}
.input-field { padding-left: 40px; }
```

## D11. Forms are prefilled — POLISH

Every field the app already knows the answer to is friction: the logged-in user's name and email on a support form, the current value on an edit form, the invite's email on the signup it leads to.

**Hunt for:** edit forms whose `defaultValue`s are empty; signup pages that ignore an `?email=` param the invite link carries.

## D12. Unsaved work survives an accident — POLISH

A long form (an editor, a multi-step flow, a settings page) that loses everything on an accidental back-swipe or refresh. Warn on `beforeunload` while the form is dirty, or persist the draft to `sessionStorage`.

**Hunt for:** multi-step flows holding all state in one component's `useState`; rich-text editors with no autosave or dirty guard.

## D13. Walk each critical form by hand — always

From code you can prove the attributes. You cannot prove the flow. Put on the device list, per critical form: "complete signup using only the keyboard", "submit with the network offline and confirm the error shows and the fields keep their values", "double-click Pay and confirm one request in the network tab", "fill the form with the browser's autofill and a password manager", "open it on an iPhone and confirm no field zooms the page".
