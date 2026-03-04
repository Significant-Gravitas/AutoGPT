"use client";

/**
 * Adapted from AI SDK Elements `prompt-input` component.
 * @see https://elements.ai-sdk.dev/components/prompt-input
 *
 * Stripped down to only the sub-components used by the copilot ChatInput:
 * PromptInput, PromptInputBody, PromptInputTextarea, PromptInputFooter,
 * PromptInputTools, PromptInputButton, PromptInputSubmit.
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
  InputGroupAddon,
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
  ArrowUp as ArrowUpIcon,
  Stop as StopIcon,
} from "@phosphor-icons/react";
import { Children, useCallback, useEffect, useRef, useState } from "react";

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
// PromptInputBody — content wrapper
// ============================================================================

export type PromptInputBodyProps = HTMLAttributes<HTMLDivElement>;

export function PromptInputBody({ className, ...props }: PromptInputBodyProps) {
  return <div className={cn("contents", className)} {...props} />;
}

// ============================================================================
// PromptInputTextarea — auto-resize textarea with Enter-to-submit
// ============================================================================

export type PromptInputTextareaProps = ComponentProps<
  typeof InputGroupTextarea
>;

export function PromptInputTextarea({
  onKeyDown,
  onChange,
  className,
  placeholder = "Type your message...",
  value,
  ...props
}: PromptInputTextareaProps) {
  const [isComposing, setIsComposing] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  function autoResize(el: HTMLTextAreaElement) {
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }

  // Resize when value changes externally (e.g. cleared after send)
  useEffect(() => {
    if (textareaRef.current) autoResize(textareaRef.current);
  }, [value]);

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

        const { form } = e.currentTarget;
        const submitButton = form?.querySelector(
          'button[type="submit"]',
        ) as HTMLButtonElement | null;
        if (submitButton?.disabled) return;

        form?.requestSubmit();
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
// PromptInputFooter — bottom bar
// ============================================================================

export type PromptInputFooterProps = Omit<
  ComponentProps<typeof InputGroupAddon>,
  "align"
>;

export function PromptInputFooter({
  className,
  ...props
}: PromptInputFooterProps) {
  return (
    <InputGroupAddon
      align="block-end"
      className={cn("justify-between gap-1", className)}
      {...props}
    />
  );
}

// ============================================================================
// PromptInputTools — left-side button group
// ============================================================================

export type PromptInputToolsProps = HTMLAttributes<HTMLDivElement>;

export function PromptInputTools({
  className,
  ...props
}: PromptInputToolsProps) {
  return (
    <div
      className={cn("flex min-w-0 items-center gap-1", className)}
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

  let Icon = <ArrowUpIcon className="size-4" weight="bold" />;

  if (status === "submitted") {
    Icon = <Spinner />;
  } else if (status === "streaming") {
    Icon = <StopIcon className="size-4" weight="bold" />;
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
