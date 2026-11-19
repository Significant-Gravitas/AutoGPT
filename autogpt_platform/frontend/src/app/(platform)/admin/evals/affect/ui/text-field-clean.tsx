"use client";

import {
  useEffect,
  useId,
  useRef,
  useState,
  type ComponentPropsWithRef,
  type ReactNode,
} from "react";

/*
 * Text Field – a single self-contained file, clean style.
 * One input for every form: a label, an optional hint, prefix and suffix text or icons, a clear, reveal or copy
 * button, a character count and an error that takes the hint's place. White with a hairline shadow; focus only
 * firms up the edge – no ring. The styles are injected below, so no extra files are needed.
 */

export type TextFieldProps = Omit<
  ComponentPropsWithRef<"input">,
  "size" | "prefix"
> & {
  /** Shown above the field and tied to it */
  label?: ReactNode;
  /** A quiet line under the field – the error takes its place while there is one */
  hint?: ReactNode;
  /** Marks the field invalid and shows this under it, in place of the hint */
  error?: ReactNode;
  /** Text or an icon inside the field, before the value – e.g. "https://" or a search icon */
  prefix?: ReactNode;
  /** Text or an icon inside the field, after the value – e.g. ".acme.app" or "USD" */
  suffix?: ReactNode;
  /** sm 28 · md 32 · lg 36 high */
  size?: "sm" | "md" | "lg";
  /** "Optional" next to the label */
  optional?: boolean;
  /** A small × that empties the field while it has a value */
  clearable?: boolean;
  /** A copy button – for API keys, links and IDs, usually with readOnly */
  copyable?: boolean;
  /** Shows "12/40" under the field; needs maxLength */
  counter?: boolean;
  /** Called after the × emptied the field (onChange fires too) */
  onClear?: () => void;
  /** Extra class on the input itself; className goes on the wrapper */
  inputClassName?: string;
};

export function TextField({
  label,
  hint,
  error,
  prefix,
  suffix,
  size = "md",
  optional = false,
  clearable = false,
  copyable = false,
  counter = false,
  onClear,
  onKeyDown,
  className,
  inputClassName,
  id,
  type = "text",
  value,
  defaultValue,
  onChange,
  disabled,
  readOnly,
  maxLength,
  ref,
  "aria-describedby": describedBy,
  ...rest
}: TextFieldProps) {
  const autoId = useId();
  const inputId = id ?? `${autoId}-input`;
  const messageId = `${autoId}-message`;
  const counterId = `${autoId}-count`;
  const input = useRef<HTMLInputElement | null>(null);

  // The current text, for the count and the × – follows value when controlled, the input itself otherwise
  const [own, setOwn] = useState(String(defaultValue ?? ""));
  const text = value !== undefined && value !== null ? String(value) : own;

  const isPassword = type === "password";
  const [revealed, setRevealed] = useState(false);
  const [copied, setCopied] = useState(false);
  const copyTimer = useRef(0);
  useEffect(() => () => window.clearTimeout(copyTimer.current), []);

  const invalid =
    Boolean(error) ||
    rest["aria-invalid"] === true ||
    rest["aria-invalid"] === "true";
  // Only text or elements count – error={true} marks the field invalid without leaving an empty line
  const message = shows(error) ? error : shows(hint) ? hint : null;
  const showCount = counter && typeof maxLength === "number";
  // Without a label, the placeholder names the field for screen readers
  const fallbackName =
    label == null &&
    !rest["aria-label"] &&
    !rest["aria-labelledby"] &&
    typeof rest.placeholder === "string"
      ? rest.placeholder
      : undefined;

  // An uncontrolled field follows form.reset(), so the count and the × don't show the old text
  useEffect(() => {
    const form = input.current?.form;
    if (!form || value !== undefined) return;
    const onReset = () =>
      window.setTimeout(() => setOwn(input.current?.value ?? ""));
    form.addEventListener("reset", onReset);
    return () => form.removeEventListener("reset", onReset);
  }, [value]);
  const showClear = clearable && !disabled && !readOnly && text.length > 0;
  const scale = SIZES.includes(size) ? size : "md";

  const clear = () => {
    const el = input.current;
    if (!el) return;
    // Sets the value the way typing would, so React's onChange runs for controlled fields too
    Object.getOwnPropertyDescriptor(
      HTMLInputElement.prototype,
      "value",
    )?.set?.call(el, "");
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.focus();
    onClear?.();
  };

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // No clipboard (an insecure page or a denied permission): select the text, so ⌘C works
      input.current?.select();
      return;
    }
    setCopied(true);
    window.clearTimeout(copyTimer.current);
    copyTimer.current = window.setTimeout(() => setCopied(false), 1600);
  };

  return (
    <div
      className={["ui-tfc", className].filter(Boolean).join(" ")}
      data-size={scale}
      data-invalid={invalid || undefined}
      data-disabled={disabled || undefined}
      data-readonly={readOnly || undefined}
    >
      {/* React 19 hoists this into <head> once, no matter how many fields render */}
      <style href="ui-text-field-clean" precedence="default">
        {css}
      </style>
      {label != null && (
        <div className="ui-tfc__head">
          <label className="ui-tfc__label" htmlFor={inputId}>
            {label}
          </label>
          {optional && <span className="ui-tfc__optional">Optional</span>}
        </div>
      )}
      {/* A click anywhere on the box – the prefix, the padding – puts the caret in the input */}
      <div
        className="ui-tfc__box"
        onPointerDown={(e) => {
          if (
            e.target === input.current ||
            (e.target as HTMLElement).closest("button")
          )
            return;
          e.preventDefault();
          input.current?.focus();
        }}
      >
        {prefix != null && (
          <span className="ui-tfc__affix" data-side="start">
            {/* Long text is cut at its start – the part next to the value is the one that matters */}
            {typeof prefix === "string" ? (
              <span className="ui-tfc__affix-text">{prefix}</span>
            ) : (
              prefix
            )}
          </span>
        )}
        <input
          {...rest}
          ref={(el) => {
            input.current = el;
            if (typeof ref === "function") ref(el);
            else if (ref) ref.current = el;
          }}
          id={inputId}
          className={["ui-tfc__input", inputClassName]
            .filter(Boolean)
            .join(" ")}
          type={isPassword && revealed ? "text" : type}
          value={value}
          defaultValue={defaultValue}
          disabled={disabled}
          readOnly={readOnly}
          maxLength={maxLength}
          aria-label={rest["aria-label"] ?? fallbackName}
          aria-invalid={invalid || undefined}
          aria-describedby={
            [
              describedBy,
              message ? messageId : null,
              showCount ? counterId : null,
            ]
              .filter(Boolean)
              .join(" ") || undefined
          }
          dir={rest.dir ?? "auto"}
          data-revealed={isPassword ? (revealed ? "true" : "false") : undefined}
          onKeyDown={(e) => {
            // Esc empties a clearable field, like a search field on macOS and iOS
            if (e.key === "Escape" && showClear) {
              e.preventDefault();
              clear();
            }
            onKeyDown?.(e);
          }}
          onChange={(e) => {
            if (value === undefined) setOwn(e.target.value);
            onChange?.(e);
          }}
        />
        {suffix != null && (
          <span className="ui-tfc__affix" data-side="end">
            {suffix}
          </span>
        )}
        {showClear && (
          <button
            type="button"
            className="ui-tfc__action"
            aria-label="Clear"
            tabIndex={-1}
            onClick={clear}
          >
            <ClearIcon />
          </button>
        )}
        {isPassword && !disabled && (
          <button
            type="button"
            className="ui-tfc__action"
            aria-label={revealed ? "Hide password" : "Show password"}
            aria-pressed={revealed}
            onClick={() => setRevealed((r) => !r)}
          >
            <span className="ui-tfc__swap" key={revealed ? "hide" : "show"}>
              {revealed ? <EyeOffIcon /> : <EyeIcon />}
            </span>
          </button>
        )}
        {copyable && (
          <button
            type="button"
            className="ui-tfc__action"
            data-copied={copied || undefined}
            aria-label={copied ? "Copied" : "Copy"}
            onClick={copy}
            disabled={disabled || !text}
          >
            <span className="ui-tfc__swap" key={copied ? "done" : "copy"}>
              {copied ? <CheckIcon /> : <CopyIcon />}
            </span>
          </button>
        )}
        {copyable && (
          <span className="ui-tfc__sr" role="status">
            {copied ? "Copied" : ""}
          </span>
        )}
      </div>
      {(message || showCount) && (
        <div className="ui-tfc__foot">
          {/* The error swaps in where the hint was, so the form below doesn't move */}
          <p
            className="ui-tfc__message"
            id={messageId}
            key={shows(error) ? "error" : "hint"}
            dir="auto"
            aria-live={shows(error) ? "polite" : undefined}
          >
            {message}
          </p>
          {showCount && (
            <span
              className="ui-tfc__count"
              id={counterId}
              data-full={text.length === maxLength || undefined}
              data-over={text.length > maxLength || undefined}
            >
              {text.length}/{maxLength}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

const SIZES = ["sm", "md", "lg"];
/** Text or an element – not true, false, null or an empty string */
const shows = (v: ReactNode) => v != null && typeof v !== "boolean" && v !== "";

/* Icons: Hugeicons Free (MIT), drawn at 16px */
function ClearIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="16"
      height="16"
      fill="currentColor"
      aria-hidden="true"
    >
      <path d="M12 2.25c-5.385 0-9.75 4.365-9.75 9.75s4.365 9.75 9.75 9.75 9.75-4.365 9.75-9.75S17.385 2.25 12 2.25Zm3.53 12.22a.75.75 0 1 1-1.06 1.06L12 13.06l-2.47 2.47a.75.75 0 0 1-1.06-1.06L10.94 12 8.47 9.53a.75.75 0 0 1 1.06-1.06L12 10.94l2.47-2.47a.75.75 0 1 1 1.06 1.06L13.06 12l2.47 2.47Z" />
    </svg>
  );
}
function EyeIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M21.544 11.045C21.848 11.471 22 11.685 22 12c0 .316-.152.529-.456.955C20.178 14.871 16.689 19 12 19c-4.69 0-8.178-4.13-9.544-6.045C2.152 12.529 2 12.315 2 12c0-.316.152-.529.456-.955C3.822 9.129 7.311 5 12 5c4.69 0 8.178 4.13 9.544 6.045Z" />
      <path d="M15 12a3 3 0 1 0-6 0 3 3 0 0 0 6 0Z" />
    </svg>
  );
}
function EyeOffIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M19.439 15.439c.996-.91 1.719-1.82 2.105-2.36.304-.426.456-.64.456-.955 0-.316-.152-.529-.456-.955C20.178 9.253 16.689 5.124 12 5.124c-.908 0-1.77.155-2.582.418M6.74 6.869c-2.09 1.267-3.6 3.083-4.284 4.04-.304.426-.456.64-.456.955 0 .316.152.529.456.955C3.822 14.735 7.311 18.864 12 18.864c2.008 0 3.786-.757 5.26-1.746" />
      <path d="M9.858 10a2.98 2.98 0 0 0-.858 2.124 3 3 0 0 0 5.124 2.12M3 3l18 18" />
    </svg>
  );
}
function CopyIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M9 15c0-2.828 0-4.243.879-5.121C10.757 9 12.172 9 15 9h1c2.828 0 4.243 0 5.121.879C22 10.757 22 12.172 22 15v1c0 2.828 0 4.243-.879 5.121C20.243 22 18.828 22 16 22h-1c-2.828 0-4.243 0-5.121-.879C9 20.243 9 18.828 9 16v-1Z" />
      <path d="M17 9c-.003-2.957-.047-4.489-.908-5.538a4 4 0 0 0-.554-.554C14.43 2 12.788 2 9.5 2c-3.287 0-4.931 0-6.038.908a4 4 0 0 0-.554.554C2 4.57 2 6.212 2 9.5c0 3.287 0 4.931.908 6.038a4 4 0 0 0 .554.554c1.05.86 2.58.906 5.538.908" />
    </svg>
  );
}
function CheckIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path className="ui-tfc__check" d="M5 12.5 9.5 17 19 7" pathLength={1} />
    </svg>
  );
}

const css = /* css */ `
/* Colors and motion as variables – override them on .ui-tfc (see Theming in the docs) */
.ui-tfc {
  --ui-tfc-title: var(--aui-ink, rgba(0, 0, 0, 0.88));
  --ui-tfc-ink: var(--aui-ink, rgba(0, 0, 0, 0.88));
  --ui-tfc-muted: var(--aui-muted, rgba(0, 0, 0, 0.61));
  --ui-tfc-subtle: var(--aui-muted, rgba(0, 0, 0, 0.45));
  --ui-tfc-faint: var(--aui-placeholder, rgba(0, 0, 0, 0.3));
  --ui-tfc-icon: var(--aui-muted, rgba(0, 0, 0, 0.447));
  --ui-tfc-surface: var(--aui-surface, #ffffff);
  --ui-tfc-wash: rgba(0, 0, 0, 0.04);
  --ui-tfc-readonly: rgba(0, 0, 0, 0.02);
  --ui-tfc-shadow: 0 0 0 0.5px var(--aui-border, rgba(0, 0, 0, 0.08)), 0 1px 1px -1px rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  --ui-tfc-shadow-hover: 0 0 0 0.5px var(--aui-border-strong, rgba(0, 0, 0, 0.13)), 0 1px 1px -1px rgba(0, 0, 0, 0.12), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
  --ui-tfc-shadow-focus: 0 0 0 1px var(--aui-border-strong, rgba(0, 0, 0, 0.28)), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
  --ui-tfc-danger: var(--aui-danger, #c4320a);
  --ui-tfc-danger-edge: color-mix(in srgb, var(--aui-red, #f04438) 55%, transparent);
  --ui-tfc-danger-edge-focus: rgba(229, 72, 77, 0.8);
  --ui-tfc-success: var(--aui-success, #1e8a4f);
  --ui-tfc-focus: var(--aui-focus, rgba(0, 96, 226, 0.7));
  --ui-tfc-ease: cubic-bezier(0.22, 1, 0.36, 1);

  --h: 32px;
  --px: 10px;
  --fs: 13px;
  --radius: var(--aui-radius, 7px);

  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
  font-family: var(--aui-font, "Inter", "InterVariable", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif);
  font-size: 13px;
  line-height: 20px;
  color: var(--ui-tfc-ink);
  -webkit-font-smoothing: antialiased;
}
.ui-tfc[data-size="sm"] { --h: 28px; --px: 8px; --fs: 13px; --radius: calc(var(--aui-radius, 7px) - 1px); }
.ui-tfc[data-size="lg"] { --h: 36px; --px: 12px; --fs: 14px; --radius: calc(var(--aui-radius, 7px) + 1px); }
/* Touch screens: a comfortable tap height, and 16px text so iOS doesn't zoom in on focus */
@media (pointer: coarse) {
  .ui-tfc { --fs: 16px; }
  .ui-tfc[data-size="sm"] { --h: 34px; }
  .ui-tfc[data-size="md"] { --h: 38px; }
  .ui-tfc[data-size="lg"] { --h: 40px; }
}

.ui-tfc__head { display: flex; align-items: baseline; gap: 6px; min-width: 0; }
.ui-tfc__label { min-width: 0; font-weight: 500; color: var(--ui-tfc-title); overflow-wrap: anywhere; cursor: default; }
.ui-tfc__optional { flex: none; font-size: 12px; line-height: 16px; color: var(--ui-tfc-subtle); }

/* The box: white, a hairline shadow as its edge; hover and focus only make that edge firmer */
.ui-tfc__box {
  display: flex;
  align-items: center;
  gap: 6px;
  box-sizing: border-box;
  height: var(--h);
  min-width: 0;
  padding-block: 0;
  padding-inline: var(--px) calc(var(--px) - 4px);
  border-radius: var(--radius);
  background: var(--ui-tfc-surface);
  box-shadow: var(--ui-tfc-shadow);
  cursor: text;
  transition: box-shadow 140ms ease, background-color 140ms ease;
}
/* Without a button at the end, both sides sit at the same distance */
.ui-tfc__box:not(:has(.ui-tfc__action)) { padding-inline-end: var(--px); }
.ui-tfc__box:hover { box-shadow: var(--ui-tfc-shadow-hover); }
.ui-tfc__box:focus-within { box-shadow: var(--ui-tfc-shadow-focus); }
.ui-tfc[data-invalid] .ui-tfc__box { box-shadow: 0 0 0 1px var(--ui-tfc-danger-edge), 0 1px 2px 0 rgba(0, 0, 0, 0.05); }
.ui-tfc[data-invalid] .ui-tfc__box:focus-within { box-shadow: 0 0 0 1px var(--ui-tfc-danger-edge-focus), 0 1px 2px 0 rgba(0, 0, 0, 0.05); }
.ui-tfc[data-readonly] .ui-tfc__box { background: var(--ui-tfc-readonly); box-shadow: 0 0 0 0.5px rgba(0, 0, 0, 0.08); cursor: default; }
.ui-tfc[data-disabled] .ui-tfc__box { background: var(--ui-tfc-readonly); box-shadow: 0 0 0 0.5px rgba(0, 0, 0, 0.06); cursor: not-allowed; }
.ui-tfc[data-disabled] { opacity: 0.55; }
.ui-tfc[data-invalid] .ui-tfc__box { animation: ui-tfc-nudge 320ms var(--ui-tfc-ease); }

.ui-tfc__input {
  flex: 1;
  min-width: 0;
  height: 100%;
  margin: 0;
  padding: 0;
  border: 0;
  outline: 0;
  background: transparent;
  font: inherit;
  font-size: var(--fs);
  color: var(--ui-tfc-ink);
  text-overflow: ellipsis;
  -webkit-appearance: none;
  appearance: none;
}
/* Next to a prefix or suffix the value keeps at least 6em – they shrink first */
.ui-tfc__box:has(> .ui-tfc__affix) .ui-tfc__input { flex: 1 0 6em; min-width: 6em; }
.ui-tfc__input::placeholder { color: var(--ui-tfc-faint); opacity: 1; }
.ui-tfc__input:disabled { cursor: not-allowed; }
.ui-tfc__input[type="search"]::-webkit-search-cancel-button { display: none; }
.ui-tfc__input[type="number"] { -moz-appearance: textfield; font-variant-numeric: tabular-nums; }
.ui-tfc__input[type="number"]::-webkit-inner-spin-button,
.ui-tfc__input[type="number"]::-webkit-outer-spin-button { -webkit-appearance: none; margin: 0; }
/* Revealing a password: the characters come out of a short blur */
.ui-tfc__input[data-revealed] { animation: ui-tfc-reveal 260ms var(--ui-tfc-ease); }
.ui-tfc__input[data-revealed="false"] { animation-name: ui-tfc-reveal-hide; }

/* Prefix and suffix: quiet text or an icon inside the box */
.ui-tfc__affix {
  display: inline-flex;
  flex: 0 1 auto;
  align-items: center;
  min-width: 0;
  max-width: 70%;
  font-size: var(--fs);
  color: var(--ui-tfc-subtle);
  white-space: nowrap;
  user-select: none;
  overflow: hidden;
  text-overflow: ellipsis;
}
/* Text right against the value, as if it were part of it ("https://" + "acme") */
.ui-tfc__affix[data-side="start"] { margin-inline-end: -6px; }
.ui-tfc__affix[data-side="end"] { max-width: 40%; margin-inline-start: -6px; }
/* A long prefix loses its start: "…/workspaces/" stays next to the value */
.ui-tfc__affix-text { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ui-tfc__affix[data-side="start"]:dir(ltr) .ui-tfc__affix-text { direction: rtl; text-align: left; }
.ui-tfc__affix[data-side="start"]:dir(ltr) .ui-tfc__affix-text::after { content: "\\200E"; }
/* An icon keeps its distance */
.ui-tfc__affix[data-side]:has(> svg) { margin: 0; color: var(--ui-tfc-icon); }
.ui-tfc__affix > svg { display: block; width: 16px; height: 16px; flex: none; }

/* Clear, reveal and copy: small ghost buttons at the end */
.ui-tfc__action {
  display: grid;
  place-items: center;
  flex: none;
  width: calc(var(--h) - 8px);
  height: calc(var(--h) - 8px);
  position: relative;
  max-width: 26px;
  max-height: 26px;
  margin: 0;
  padding: 0;
  border: 0;
  border-radius: 5px;
  background: transparent;
  color: var(--ui-tfc-icon);
  cursor: pointer;
  -webkit-tap-highlight-color: transparent;
  transition: background-color 140ms ease, color 140ms ease;
}
.ui-tfc__action + .ui-tfc__action { margin-left: -4px; }
/* Touch screens: bigger buttons and a hit area that reaches past them */
@media (pointer: coarse) {
  .ui-tfc__action { width: 32px; height: 32px; max-width: none; max-height: none; }
  .ui-tfc__action::before { content: ""; position: absolute; inset: -4px; }
}
.ui-tfc__action:hover:not(:disabled) { background: var(--ui-tfc-wash); color: var(--ui-tfc-ink); }
.ui-tfc__action:focus-visible { outline: 2px solid var(--ui-tfc-focus); outline-offset: -2px; }
.ui-tfc__action:disabled { opacity: 0.4; cursor: default; }
.ui-tfc__action[data-copied] { color: var(--ui-tfc-success); }
.ui-tfc__action svg { display: block; }
.ui-tfc__swap { display: grid; animation: ui-tfc-swap 280ms var(--ui-tfc-ease); }
.ui-tfc__check { stroke-dasharray: 1; animation: ui-tfc-draw 320ms 60ms var(--ui-tfc-ease) both; }

/* Under the field: the hint, or the error in its place, and the count */
.ui-tfc__foot { display: flex; align-items: flex-start; gap: 12px; min-width: 0; }
.ui-tfc__message {
  flex: 1;
  min-width: 0;
  margin: 0;
  font-size: 12px;
  line-height: 16px;
  color: var(--ui-tfc-subtle);
  overflow-wrap: anywhere;
  animation: ui-tfc-message 260ms var(--ui-tfc-ease);
}
.ui-tfc[data-invalid] .ui-tfc__message { color: var(--ui-tfc-danger); }
.ui-tfc__count { flex: none; margin-left: auto; font-size: 12px; line-height: 16px; font-variant-numeric: tabular-nums; color: var(--ui-tfc-faint); transition: color 140ms ease; }
.ui-tfc__count[data-over] { color: var(--ui-tfc-danger); }
.ui-tfc__sr { position: absolute; width: 1px; height: 1px; margin: -1px; padding: 0; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; border: 0; }
.ui-tfc__count[data-full] { color: var(--ui-tfc-muted); }

@keyframes ui-tfc-nudge {
  30% { transform: translateX(-3px); }
  65% { transform: translateX(2px); }
}
@keyframes ui-tfc-message {
  from { opacity: 0; transform: translateY(-3px); filter: blur(2px); }
}
@keyframes ui-tfc-reveal {
  from { filter: blur(4px); opacity: 0.4; }
}
@keyframes ui-tfc-reveal-hide {
  from { filter: blur(3px); opacity: 0.5; }
}
@keyframes ui-tfc-swap {
  from { opacity: 0; transform: scale(0.6); filter: blur(2px); }
}
@keyframes ui-tfc-draw {
  from { stroke-dashoffset: 1; }
  to { stroke-dashoffset: 0; }
}

@media (prefers-reduced-motion: reduce) {
  .ui-tfc *, .ui-tfc[data-invalid] .ui-tfc__box { animation-duration: 1ms !important; }
  .ui-tfc__box { transition-duration: 1ms; }
}

/* Dark mode: follows a .dark class or data-theme="dark" on any ancestor, e.g. <html> */
:where(.dark, [data-theme="dark"]) .ui-tfc {
  --ui-tfc-title: var(--aui-ink, rgba(255, 255, 255, 0.92));
  --ui-tfc-ink: var(--aui-ink, rgba(255, 255, 255, 0.9));
  --ui-tfc-muted: var(--aui-muted, rgba(255, 255, 255, 0.62));
  --ui-tfc-subtle: var(--aui-muted, rgba(255, 255, 255, 0.46));
  --ui-tfc-faint: var(--aui-placeholder, rgba(255, 255, 255, 0.3));
  --ui-tfc-icon: var(--aui-muted, rgba(255, 255, 255, 0.5));
  --ui-tfc-surface: var(--aui-surface, #1c1c1e);
  --ui-tfc-wash: rgba(255, 255, 255, 0.07);
  --ui-tfc-readonly: rgba(255, 255, 255, 0.03);
  --ui-tfc-shadow: 0 0 0 0.5px var(--aui-border, rgba(255, 255, 255, 0.14)), 0 1px 2px 0 rgba(0, 0, 0, 0.4);
  --ui-tfc-shadow-hover: 0 0 0 0.5px var(--aui-border-strong, rgba(255, 255, 255, 0.2)), 0 1px 2px 0 rgba(0, 0, 0, 0.4);
  --ui-tfc-shadow-focus: 0 0 0 1px var(--aui-border-strong, rgba(255, 255, 255, 0.3)), 0 1px 2px 0 rgba(0, 0, 0, 0.4);
  --ui-tfc-danger: var(--aui-danger, #ff8a7a);
  --ui-tfc-danger-edge: color-mix(in srgb, var(--aui-red, #ff6b5f) 50%, transparent);
  --ui-tfc-danger-edge-focus: rgba(255, 99, 105, 0.75);
  --ui-tfc-success: var(--aui-success, #7dcf9f);
  --ui-tfc-focus: var(--aui-focus, rgba(77, 141, 255, 0.8));
}
:where(.dark, [data-theme="dark"]) .ui-tfc[data-readonly] .ui-tfc__box,
:where(.dark, [data-theme="dark"]) .ui-tfc[data-disabled] .ui-tfc__box { box-shadow: 0 0 0 0.5px rgba(255, 255, 255, 0.12); }
`;
