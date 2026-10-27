import { isComposingEvent } from "@/lib/keyboard";
import { cn } from "@/lib/utils";
import { forwardRef, ReactNode, useState } from "react";
import CurrencyInput from "react-currency-input-field";
import { Text } from "../Text/Text";
import type { Variant } from "../Text/helpers";
import { InformationTooltip } from "@/components/molecules/InformationTooltip/InformationTooltip";
import { BaseInput, type BaseInputProps } from "./BaseInput";
import { useInput } from "./useInput";
import {
  BareInput,
  BareTextarea,
  type BareInputProps,
  type BareTextareaProps,
} from "./BareInput";
import { EyeIcon, EyeOffIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

type InputElement = HTMLInputElement | HTMLTextAreaElement;

export interface TextFieldProps
  extends Omit<BaseInputProps, "size" | "onKeyDown"> {
  label: string;
  id: string;
  hideLabel?: boolean;
  decimalCount?: number; // Only used for type="amount"
  error?: string;
  hint?: ReactNode;
  size?: "small" | "medium";
  labelVariant?: Variant;
  labelClassName?: string;
  labelTooltip?: string;
  wrapperClassName?: string;
  type?:
    | "text"
    | "email"
    | "password"
    | "number"
    | "amount"
    | "tel"
    | "url"
    | "textarea"
    | "date"
    | "datetime-local";
  // Widened over BaseInputProps, which only describes the <input> branch, so the
  // handler is callable with either element's event and needs no cast.
  onKeyDown?: React.KeyboardEventHandler<InputElement>;
  // Textarea-specific props
  rows?: number;
  amountPrefix?: string;
  amountSuffix?: string;
  /** `body-md` sets the value at the text-body-md role (15px/1.5rem) for
   *  sizes whose text is text-body, e.g. inside a profile tab. */
  textSize?: "body" | "body-md";
}

// `<Input bare>` renders only the element, as the deleted components/ui/input
// and textarea did; anything else is the labelled field. The field's
// signature comes last, so ComponentProps<typeof Input> stays the field's.
export const Input = forwardRef<
  InputElement,
  TextFieldProps | BareInputProps | BareTextareaProps
>(function Input(props, ref) {
  if ("bare" in props && props.bare) {
    if (props.type === "textarea") {
      return (
        <BareTextarea
          {...(props as BareTextareaProps)}
          ref={ref as React.Ref<HTMLTextAreaElement>}
        />
      );
    }
    return (
      <BareInput
        {...(props as BareInputProps)}
        ref={ref as React.Ref<HTMLInputElement>}
      />
    );
  }
  return <TextField {...(props as TextFieldProps)} ref={ref} />;
}) as unknown as typeof BareTextarea & typeof BareInput & typeof TextField;

const TextField = forwardRef<InputElement, TextFieldProps>(function TextField(
  {
    className,
    label,
    placeholder,
    hideLabel = false,
    decimalCount,
    hint,
    error,
    size = "medium",
    labelVariant = "large-medium",
    labelClassName,
    labelTooltip,
    wrapperClassName,
    amountPrefix,
    amountSuffix,
    onKeyDown,
    textSize = "body",
    ...props
  },
  ref,
) {
  // Consumers never see keydowns an IME is still composing; see AGENTS.md
  // "Keyboard handling". Left undefined when the consumer passes no handler, so
  // we don't attach a listener that does nothing.
  function handleKeyDown(e: React.KeyboardEvent<InputElement>) {
    if (isComposingEvent(e)) return;
    onKeyDown?.(e);
  }
  const guardedOnKeyDown = onKeyDown ? handleKeyDown : undefined;

  const { handleInputChange, handleTextareaChange, handleAmountValueChange } =
    useInput({
      type: props.type,
      onChange: props.onChange,
      decimalCount,
    });
  const [showPassword, setShowPassword] = useState(false);
  const inputId = props.id;

  const isPasswordType = props.type === "password";
  const inputType = isPasswordType && showPassword ? "text" : props.type;

  function handleTogglePassword() {
    setShowPassword((prev) => !prev);
  }

  function handleWrapperBlur(e: React.FocusEvent<HTMLDivElement>) {
    if (!e.currentTarget.contains(e.relatedTarget)) {
      setShowPassword(false);
    }
  }

  const baseStyles = cn(
    // Base styles
    "rounded-xl border border-border-base bg-surface px-4 shadow-none w-full",
    "font-normal text-foreground-base",
    "placeholder:font-normal placeholder:text-placeholder",
    // Focus and hover states
    // TODO(tokens): no role is purple-400 #925CF7, the focus border and ring.
    "focus:border-purple-400 focus:shadow-none focus:outline-none focus:ring-1 focus:ring-purple-400 focus:ring-offset-0",
    className,
  );

  const bodyTextClass = textSize === "body-md" ? "text-body-md" : "text-body";

  const errorStyles =
    error && "!border !border-error focus:border-error focus:ring-error";

  const renderInput = () => {
    if (props.type === "textarea") {
      return (
        <textarea
          ref={ref as React.Ref<HTMLTextAreaElement>}
          data-slot="input-control"
          className={cn(
            baseStyles,
            errorStyles,
            "-mb-1 h-auto min-h-[2.875rem]",
            // Size variants for textarea
            size === "small" && [
              "min-h-9", // 36px minimum
              "py-2",
              bodyTextClass,
              "placeholder:text-body",
            ],
            size === "medium" && [
              "min-h-[2.875rem]", // 46px minimum (current default)
              bodyTextClass,
              "py-2.5",
            ],
          )}
          placeholder={placeholder || label}
          onChange={handleTextareaChange}
          onKeyDown={guardedOnKeyDown}
          rows={props.rows || 3}
          aria-label={
            props["aria-label"] ?? (hideLabel && label ? label : undefined)
          }
          aria-labelledby={props["aria-labelledby"]}
          aria-describedby={props["aria-describedby"]}
          id={inputId}
          disabled={props.disabled}
          value={props.value}
          maxLength={props.maxLength}
          name={props.name}
          required={props.required}
        />
      );
    }

    if (props.type === "amount") {
      return (
        <CurrencyInput
          data-slot="input-control"
          className={cn(
            baseStyles,
            errorStyles,
            // Size variants
            size === "small" && [
              "h-9",
              "py-2",
              bodyTextClass,
              "placeholder:text-body",
            ],
            size === "medium" && [
              "h-[2.875rem]",
              "py-2.5",
              textSize === "body-md" && "text-body-md",
            ],
          )}
          placeholder={placeholder || label}
          // CurrencyInput gives unformatted numeric string in value param
          onValueChange={handleAmountValueChange}
          value={props.value as string | number | undefined}
          id={inputId}
          name={props.name}
          disabled={props.disabled}
          inputMode="decimal"
          decimalsLimit={decimalCount ?? 4}
          allowDecimals={decimalCount !== 0}
          groupSeparator=","
          decimalSeparator="."
          allowNegativeValue
          aria-label={
            props["aria-label"] ?? (hideLabel && label ? label : undefined)
          }
          aria-labelledby={props["aria-labelledby"]}
          aria-describedby={props["aria-describedby"]}
          // Pass through common handlers
          onBlur={props.onBlur as any}
          onFocus={props.onFocus as any}
          onKeyDown={guardedOnKeyDown}
          prefix={amountPrefix}
          suffix={amountSuffix}
        />
      );
    }

    return (
      <BaseInput
        ref={ref as React.Ref<HTMLInputElement>}
        data-slot="input-control"
        className={cn(
          baseStyles,
          errorStyles,
          // Add padding for password toggle button
          isPasswordType && "pr-12",
          // Size variants
          size === "small" && [
            "h-9", // 36px
            "py-2",
            bodyTextClass, // 14px font, 22px line height
            "placeholder:text-body",
          ],
          size === "medium" && [
            "h-[2.875rem]", // 46px (current default)
            "py-2.5",
            // Medium sets no size by default; body-md is opt-in.
            textSize === "body-md" && "text-body-md",
          ],
        )}
        placeholder={placeholder || label}
        onChange={handleInputChange}
        {...(hideLabel && label ? { "aria-label": label } : {})}
        {...props}
        id={inputId}
        onKeyDown={guardedOnKeyDown}
        type={inputType}
      />
    );
  };

  const input = (
    <div
      data-slot="input-wrapper"
      onBlur={isPasswordType ? handleWrapperBlur : undefined}
      className={cn("relative w-full", wrapperClassName)}
    >
      {renderInput()}
      {isPasswordType && (
        <button
          data-slot="input-password-toggle"
          type="button"
          onClick={handleTogglePassword}
          disabled={props.disabled}
          className="absolute right-4 top-1/2 -translate-y-1/2 text-foreground-subtle transition-colors hover:text-foreground-secondary disabled:cursor-not-allowed disabled:opacity-50"
          aria-label={showPassword ? "Hide password" : "Show password"}
          aria-pressed={showPassword}
          aria-controls={inputId}
        >
          {showPassword ? (
            <Icon
              data-slot="input-password-toggle-icon"
              icon={EyeIcon}
              size={16}
            />
          ) : (
            <Icon
              data-slot="input-password-toggle-icon"
              icon={EyeOffIcon}
              size={16}
            />
          )}
        </button>
      )}
    </div>
  );

  const inputWithError = (
    <div
      data-slot="input-field"
      className={cn("relative mb-6 w-full", wrapperClassName)}
    >
      {input}
      <Text
        data-slot="input-error"
        variant="small-medium"
        as="span"
        tone="error"
        className={cn(
          "absolute left-0 top-full mt-1 transition-opacity duration-200",
          error ? "opacity-100" : "opacity-0",
        )}
      >
        {error || " "}{" "}
        {/* Always render with space to maintain consistent height calculation */}
      </Text>
    </div>
  );

  return hideLabel ? (
    inputWithError
  ) : (
    <div data-slot="input" className="flex w-full flex-col gap-2">
      <div
        data-slot="input-header"
        className="flex items-center justify-between gap-2"
      >
        <div data-slot="input-label-group" className="flex items-center gap-1">
          <label data-slot="input-label" htmlFor={inputId}>
            <Text
              data-slot="input-label-text"
              variant={labelVariant}
              as="span"
              className={cn("text-foreground-base", labelClassName)}
            >
              {label}
            </Text>
          </label>
          {labelTooltip ? (
            <InformationTooltip description={labelTooltip} iconSize={20} />
          ) : null}
        </div>
        {hint ? (
          <Text data-slot="input-hint" variant="small" as="span" tone="subtle">
            {hint}
          </Text>
        ) : null}
      </div>
      {inputWithError}
    </div>
  );
});
