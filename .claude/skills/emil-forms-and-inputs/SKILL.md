---
name: emil-forms-and-inputs
description: "Build forms, inputs, buttons, and controls that feel right — label association, 16px inputs to stop iOS zoom, focus states, validation timing, loading/disabled button states, Enter and Cmd+Enter submission, and autofocus rules. Use when building or reviewing any form, text input, textarea, search field, login/signup flow, settings page, checkout, button, checkbox, or submit handler. Triggers on: form, input, textarea, select, button, checkbox, radio, toggle, submit, onSubmit, validation, error message, placeholder, label, autofocus, autocomplete, spellcheck, iOS zoom, font-size 16px, focus ring, focus-visible, loading state, disabled, double submit, duplicate request, Enter to submit, Cmd+Enter, keyboard shortcut, destructive action, confirmation, delete, clear button, input icon, prefix, suffix, search field, login form, signup form, 1Password, password manager."
---

# Forms & Inputs

Forms are where users do real work, and they notice every rough edge: the label that doesn't focus the input, the page that zooms on iOS, the button that fires twice, the Enter key that does nothing. None of these are hard to fix — they're just easy to forget. Wire up the semantics the platform gives you (`<form>`, `<label>`, `<button>`), then layer on the details below. Where CSS is involved, implement it in the project's existing styling system (Tailwind, CSS Modules, styled-components — whatever is already there); the values matter, the syntax doesn't.

## Core Principles

### 1. Every input gets an associated label

Clicking the label must focus the input:

```html
<label for="email">Email</label>
<input id="email" type="email" />

<!-- Or wrap the input -->
<label>
  Email
  <input type="email" />
</label>
```

### 2. Use the right input type

```html
<input type="email" />    <input type="password" />
<input type="tel" />      <input type="url" />
<input type="number" />   <input type="search" />
```

This gets you the right mobile keyboard, browser validation, and autofill behavior for free.

### 3. Input font size ≥ 16px, always

Inputs smaller than 16px cause iOS Safari to zoom in on focus. Set it globally:

```css
input, textarea, select {
  font-size: 16px;
}
```

If the design calls for smaller text, keep the computed font-size at 16px on touch devices and scale visually some other way — never below 16px.

### 4. Visible focus states

Every input and control needs a clear focus state. Style `:focus-visible` (not bare `:focus`) so keyboard users get a ring without mouse users seeing one on every click, and never `outline: none` without a replacement:

```css
input:focus-visible { outline: 2px solid var(--focus-color); outline-offset: 2px; }
```

### 5. Kill noisy browser behaviors when they don't help

Disable `spellcheck` and `autocomplete` most of the time for cleaner UX (usernames, search fields, codes, slugs — anything that isn't a real address/identity field). Keep autocomplete ON for email, name, address, and payment fields — there it genuinely helps. Disable 1Password's overlay on fields where it doesn't belong:

```html
<input type="text" spellcheck="false" autocomplete="off" />
<input data-lpignore="true" data-1p-ignore />
```

### 6. Input decorations overlay the input, they don't sit beside it

Prefix/suffix icons and labels should be absolutely positioned on top of the input (with padding making room), not rendered as sibling elements — otherwise clicking them does nothing and the "input" border lies about the hit area:

```css
.input-wrapper {
  position: relative;
}

.input-icon {
  position: absolute;
  left: 12px;
  top: 50%;
  transform: translateY(-50%);
  pointer-events: none;
}

.input-field { padding-left: 40px; }
```

Clickable decorations (like a clear button) should refocus the input:

```jsx
<button
  className="input-icon-button"
  onClick={() => inputRef.current?.focus()}
>
  <SearchIcon />
</button>
```

### 7. Autofocus: yes in modals, never on touch

Autofocus the first input when a modal opens (if the modal contains one). Do NOT autofocus on touch devices — it pops the keyboard open unexpectedly:

```jsx
const isTouchDevice = 'ontouchstart' in window;

<input autoFocus={!isTouchDevice} />
```

### 8. Always wrap inputs in a `<form>` — Enter must submit

```jsx
<form onSubmit={handleSubmit}>
  <input type="text" />
  <button type="submit">Submit</button>
</form>
```

For textareas (where Enter inserts a newline), support `Cmd+Enter` (Mac) / `Ctrl+Enter` (Windows) to submit:

```jsx
function handleKeyDown(e) {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    handleSubmit();
  }
}
```

### 9. Buttons are `<button>` elements — with a loading state

Never put click handlers on elements that aren't buttons:

```html
<!-- Good -->
<button onClick={handleClick}>Click me</button>
<!-- Bad -->
<div onClick={handleClick}>Click me</div>
<span onClick={handleClick}>Click me</span>
```

Disable the button while submitting to prevent duplicate network requests, and say what's happening:

```jsx
const [isSubmitting, setIsSubmitting] = useState(false);

<button
  disabled={isSubmitting}
  onClick={async () => {
    setIsSubmitting(true);
    await submitForm();
    setIsSubmitting(false);
  }}
>
  {isSubmitting ? 'Submitting...' : 'Submit'}
</button>
```

Make buttons feel pressable with a subtle scale on `:active`:

```css
.button:active { transform: scale(0.97); }
```

If a button's action has a keyboard shortcut, surface it as a tooltip:

```jsx
<Tooltip content="Save (Cmd+S)">
  <button onClick={save}>Save</button>
</Tooltip>
```

### 10. Validate late, not on every keystroke

Don't flag a field as invalid while the user is still typing their first attempt — validate on blur or on submit. Once a field HAS shown an error, switch to validating on change so the error clears the moment they fix it ("reward early, punish late").

Colocate error messages with the field that caused them — not in a summary at the top:

```jsx
<div className="field">
  <input type="email" aria-invalid={!!error} />
  {error && <span className="error">{error}</span>}
</div>
```

### 11. No dead zones on checkboxes and controls

The label, the control, AND the gap between them must all be clickable:

```css
.checkbox-wrapper {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

/* Make the entire row clickable, not just the checkbox */
.checkbox-wrapper label { cursor: pointer; flex: 1; }
```

Or wrap everything in one label:

```html
<label class="checkbox-row">
  <input type="checkbox" />
  <span>Remember me</span>
</label>
```

### 12. Destructive actions require confirmation

```jsx
function handleDelete() {
  if (confirm('Are you sure you want to delete this?')) deleteItem();
}
```

`confirm()` is the floor — use a proper confirmation modal for real products.

### 13. Prefill everything you can

Use the logged-in user's data to prefill forms whenever possible. When linking to a form, prefill it from the request context — e.g. a "Change username" link should land on a form prefilled with: "I'd like to change my username to:"

### 14. Build on accessible primitives

Use Base UI for accessible component primitives. If it doesn't fit the codebase, use whatever does — but the accessibility bar (focus management, ARIA, keyboard support) is non-negotiable.

## Checklist

- [ ] Every input has a label that focuses it on click
- [ ] Correct `type` attribute on every input
- [ ] `font-size: 16px` minimum on inputs, textareas, selects
- [ ] Visible `:focus-visible` state on all controls
- [ ] `spellcheck="false"` / `autocomplete="off"` where autofill is noise
- [ ] Icons/decorations absolutely positioned over the input, clicks refocus it
- [ ] Autofocus in modals, suppressed on touch devices
- [ ] Inputs wrapped in `<form>`; Enter submits; Cmd/Ctrl+Enter submits textareas
- [ ] Real `<button>` elements; disabled + loading text during submission
- [ ] `:active { transform: scale(0.97) }` for press feedback
- [ ] Errors validate on blur/submit and render next to their field
- [ ] Checkbox rows fully clickable — no dead zones
- [ ] Destructive actions confirmed before executing
- [ ] Forms prefilled from user data and link context
- [ ] Styling written in the project's existing styling system

## Common Mistakes

- Icon rendered as a sibling of the input, so clicking it does nothing
- 14px input text that zooms the whole page on iPhone
- Submit button that stays enabled and fires the request twice
- A lone input with no `<form>`, so Enter silently does nothing
- Error summary at the top of the form instead of next to the broken field; validating on every keystroke, yelling "invalid email" after one character
- Autofocus on mobile shoving the keyboard into the user's face
- `<div onClick>` "buttons" invisible to keyboards and screen readers
