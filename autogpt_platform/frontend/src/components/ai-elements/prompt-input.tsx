"use client";

/**
 * Adapted from AI SDK Elements `prompt-input` component.
 * @see https://elements.ai-sdk.dev/components/prompt-input
 *
 * Stripped down to the sub-components the copilot ChatInput builds on:
 * PromptInput, PromptInputTextarea, PromptInputButton, PromptInputSubmit.
 * The composer lays out its own rows, so the Body/Footer/Tools wrappers no
 * longer live here.
 */

import type { ChatStatus } from "ai";
import type {
  ComponentProps,
  FormEvent,
  FormEventHandler,
  HTMLAttributes,
  KeyboardEventHandler,
  ReactNode,
} from "react";

import {
  InputGroup,
  InputGroupButton,
  InputGroupTextarea,
} from "@/components/ui/input-group";
import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  Children,
  useCallback,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { ArrowUp02Icon, StopIcon } from "@hugeicons/core-free-icons";
import { Icon as UIIcon } from "@/components/atoms/Icon/Icon";

// ============================================================================
// PromptInput — form wrapper
// ============================================================================

export type PromptInputProps = Omit<
  HTMLAttributes<HTMLFormElement>,
  "onSubmit"
> & {
  onSubmit: (
    text: string,
    event: FormEvent<HTMLFormElement>,
  ) => void | Promise<void>;
};

export function PromptInput({
  className,
  onSubmit,
  children,
  ...props
}: PromptInputProps) {
  const formRef = useRef<HTMLFormElement | null>(null);

  const handleSubmit: FormEventHandler<HTMLFormElement> = useCallback(
    async (event) => {
      event.preventDefault();
      const form = event.currentTarget;
      const formData = new FormData(form);
      const text = (formData.get("message") as string) || "";

      const result = onSubmit(text, event);
      if (result instanceof Promise) {
        await result;
      }
    },
    [onSubmit],
  );

  return (
    <form
      className={cn("w-full", className)}
      onSubmit={handleSubmit}
      ref={formRef}
      {...props}
    >
      <InputGroup className="overflow-hidden">{children}</InputGroup>
    </form>
  );
}

// ============================================================================
// PromptInputTextarea — auto-resize textarea with Enter-to-submit
// ============================================================================

export type PromptInputTextareaProps = ComponentProps<
  typeof InputGroupTextarea
> & {
  /** Reports whether the content needs more than one line of the host's
   *  single-row layout, so the host can switch to a stacked one. Derived from
   *  the box's own line-height rather than a shared pixel constant, which
   *  cannot survive hosts that restyle the textarea. */
  onMultilineChange?: (isMultiline: boolean) => void;
};

/** True once the content needs a second line. Compared against this box's own
 *  computed line-height + padding, so a host's font or padding can change
 *  without silently re-tuning the threshold. */
function isWrapped(el: HTMLTextAreaElement, contentHeight: number): boolean {
  const style = getComputedStyle(el);
  const lineHeight = parseFloat(style.lineHeight);
  if (!Number.isFinite(lineHeight) || lineHeight <= 0) return false;
  const padding =
    (parseFloat(style.paddingTop) || 0) +
    (parseFloat(style.paddingBottom) || 0);
  // One pixel of slack absorbs sub-pixel rounding.
  return contentHeight > lineHeight + padding + 1;
}

/** scrollHeight of the content alone, leaving the box at height:auto. A host
 *  min-height floors scrollHeight (the hero composer sets 4.5rem), so an empty
 *  box would report its minimum and read as already wrapped; the floor is
 *  lifted for the read and handed back. */
function measureContentHeight(el: HTMLTextAreaElement): number {
  const ownMinHeight = el.style.minHeight;
  el.style.height = "auto";
  el.style.minHeight = "0";
  const contentHeight = el.scrollHeight;
  el.style.minHeight = ownMinHeight;
  return contentHeight;
}

export function PromptInputTextarea({
  onKeyDown,
  onChange,
  className,
  placeholder = "Type your message...",
  value,
  onMultilineChange,
  ...props
}: PromptInputTextareaProps) {
  const [isComposing, setIsComposing] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  // Ref keeps autoResize stable inside the memoized change handler while
  // still reaching the latest callback.
  const onMultilineChangeRef = useRef(onMultilineChange);
  onMultilineChangeRef.current = onMultilineChange;
  // Wrapping is judged at the width the box has in the host's single row,
  // remembered from the last time it sat there. Judging it at whatever width
  // the box has right now lets the layout argue with itself: text that wraps
  // in the narrow row but fits the full-width one would flip the host between
  // the two layouts on every frame.
  const isMultilineRef = useRef(false);
  const singleRowWidthRef = useRef<number | null>(null);
  // How much of the row the host's addons take beside the box, learned the
  // first time the box is measured while stacked.
  const addonsWidthRef = useRef<number | null>(null);

  function autoResize(el: HTMLTextAreaElement) {
    const contentHeight = measureContentHeight(el);
    el.style.height = `${contentHeight}px`;

    // Border-box width, because that is what handing the number back as an
    // inline width means under `box-sizing: border-box`. `getComputedStyle`
    // reports the content box, so replaying it would silently drop the box's
    // horizontal padding and judge wrapping in a row narrower than the real
    // one. A zero width (an unmounted or collapsed row) is no measurement at
    // all: caching it would pin every later check to a row that always wraps.
    const width = el.getBoundingClientRect().width;
    const rememberedWidth = singleRowWidthRef.current;
    let wrapped: boolean;
    if (isMultilineRef.current && rememberedWidth !== null) {
      // The row itself changes while the box is stacked — a panel opening, the
      // window resizing — so the remembered width is only a starting point:
      // once the addon offset is known, the box's current full-row width gives
      // back the single-row width it would have now. Judging against the width
      // it had when it last sat in the row would leave it stacked over text
      // that now fits, or unstack it into a row it no longer fits.
      if (width > 0) {
        let singleRow = rememberedWidth;
        if (addonsWidthRef.current === null) {
          const addonsWidth = width - rememberedWidth;
          if (addonsWidth > 0) addonsWidthRef.current = addonsWidth;
        } else {
          const candidate = width - addonsWidthRef.current;
          if (candidate > 0) singleRow = candidate;
        }
        // The single row sits inside the full row, so it can never be the
        // wider of the two. Once the row has narrowed past what the addons
        // take, neither branch above can say anything, and the remembered
        // width describes a row the composer no longer has -- measuring there
        // would call text that now wraps a fit. The full row is the most the
        // box can know then.
        singleRowWidthRef.current = Math.min(singleRow, width);
      }
      const ownWidth = el.style.width;
      el.style.width = `${singleRowWidthRef.current}px`;
      wrapped = isWrapped(el, measureContentHeight(el));
      el.style.width = ownWidth;
      el.style.height = `${contentHeight}px`;
    } else {
      singleRowWidthRef.current = width > 0 ? width : null;
      // Measured from the row itself, so the offset is re-learned on the next
      // stack rather than carried over from an addon row that has since
      // changed (the connection picker hides while a turn is streaming).
      addonsWidthRef.current = null;
      wrapped = isWrapped(el, contentHeight);
    }
    if (wrapped === isMultilineRef.current) return;
    isMultilineRef.current = wrapped;
    onMultilineChangeRef.current?.(wrapped);
  }

  // Resize when value changes externally (e.g. a guided prompt dropped in,
  // or cleared after send). Runs before paint: typing resizes synchronously
  // in handleChange, but a value set from outside would otherwise paint one
  // frame at the old height — visible as a jump when the host restyles
  // itself around a now-multiline box.
  useLayoutEffect(() => {
    if (textareaRef.current) autoResize(textareaRef.current);
  }, [value]);

  // Width changes rewrap the same text — a narrowing viewport, or a panel
  // opening beside the composer — so the box must re-measure without the
  // value moving. Only width is acted on: autoResize sets the height itself,
  // so reacting to height would loop.
  useLayoutEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    let lastWidth = -1;
    const observer = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width ?? 0;
      if (width === lastWidth) return;
      lastWidth = width;
      autoResize(el);
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      autoResize(e.currentTarget);
      onChange?.(e);
    },
    [onChange],
  );

  const handleKeyDown: KeyboardEventHandler<HTMLTextAreaElement> = useCallback(
    (e) => {
      // Call external handler first
      onKeyDown?.(e);

      if (e.defaultPrevented) return;

      if (e.key === "Enter") {
        if (isComposing || e.nativeEvent.isComposing) return;
        if (e.shiftKey) return;
        e.preventDefault();

        e.currentTarget.form?.requestSubmit();
      }
    },
    [onKeyDown, isComposing],
  );

  const handleCompositionEnd = useCallback(() => setIsComposing(false), []);
  const handleCompositionStart = useCallback(() => setIsComposing(true), []);

  return (
    <InputGroupTextarea
      ref={textareaRef}
      rows={1}
      className={cn(
        "max-h-48 min-h-0 text-base leading-6 md:text-base",
        className,
      )}
      name="message"
      value={value}
      onChange={handleChange}
      onCompositionEnd={handleCompositionEnd}
      onCompositionStart={handleCompositionStart}
      onKeyDown={handleKeyDown}
      placeholder={placeholder}
      {...props}
    />
  );
}

// ============================================================================
// PromptInputButton — tool button with optional tooltip
// ============================================================================

export type PromptInputButtonTooltip =
  | string
  | {
      content: ReactNode;
      shortcut?: string;
      side?: ComponentProps<typeof TooltipContent>["side"];
    };

export type PromptInputButtonProps = ComponentProps<typeof InputGroupButton> & {
  tooltip?: PromptInputButtonTooltip;
};

export function PromptInputButton({
  variant = "ghost",
  className,
  size,
  tooltip,
  ...props
}: PromptInputButtonProps) {
  const newSize =
    size ?? (Children.count(props.children) > 1 ? "sm" : "icon-sm");

  const button = (
    <InputGroupButton
      className={cn(className)}
      size={newSize}
      type="button"
      variant={variant}
      {...props}
    />
  );

  if (!tooltip) return button;

  const tooltipContent =
    typeof tooltip === "string" ? tooltip : tooltip.content;
  const shortcut = typeof tooltip === "string" ? undefined : tooltip.shortcut;
  const side = typeof tooltip === "string" ? "top" : (tooltip.side ?? "top");

  return (
    <Tooltip>
      <TooltipTrigger asChild>{button}</TooltipTrigger>
      <TooltipContent side={side}>
        {tooltipContent}
        {shortcut && (
          <span className="ml-2 text-muted-foreground">{shortcut}</span>
        )}
      </TooltipContent>
    </Tooltip>
  );
}

// ============================================================================
// PromptInputSubmit — send / stop button
// ============================================================================

export type PromptInputSubmitProps = ComponentProps<typeof InputGroupButton> & {
  status?: ChatStatus;
  onStop?: () => void;
};

export function PromptInputSubmit({
  className,
  variant = "default",
  size = "icon-sm",
  status,
  onStop,
  onClick,
  disabled,
  children,
  ...props
}: PromptInputSubmitProps) {
  const isGenerating = status === "submitted" || status === "streaming";
  const canStop = isGenerating && Boolean(onStop);
  const isDisabled = Boolean(disabled) || (isGenerating && !canStop);

  let Icon = <UIIcon icon={ArrowUp02Icon} className="size-4" />;

  if (status === "submitted") {
    Icon = <Spinner />;
  } else if (status === "streaming") {
    Icon = <UIIcon icon={StopIcon} className="size-4" />;
  }

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLButtonElement>) => {
      if (canStop && onStop) {
        e.preventDefault();
        onStop();
        return;
      }
      if (isGenerating) {
        e.preventDefault();
        return;
      }
      onClick?.(e);
    },
    [canStop, isGenerating, onStop, onClick],
  );

  return (
    <InputGroupButton
      aria-label={canStop ? "Stop" : "Submit"}
      className={cn(
        "size-[2.625rem] rounded-full border-zinc-800 bg-zinc-800 text-white hover:border-zinc-900 hover:bg-zinc-900 disabled:border-zinc-200 disabled:bg-zinc-200 disabled:text-white disabled:opacity-100",
        className,
      )}
      disabled={isDisabled}
      onClick={handleClick}
      size={size}
      type={canStop ? "button" : "submit"}
      variant={variant}
      {...props}
    >
      {children ?? Icon}
    </InputGroupButton>
  );
}
