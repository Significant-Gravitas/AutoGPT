import {
  Input as BaseInput,
  type InputProps,
} from "@/components/__legacy__/ui/input";
import { isComposingEvent } from "@/lib/keyboard";
import { cn } from "@/lib/utils";
import { forwardRef, ReactNode, useState } from "react";
import CurrencyInput from "react-currency-input-field";
import { Text } from "../Text/Text";
import type { Variant } from "../Text/helpers";
import { InformationTooltip } from "@/components/molecules/InformationTooltip/InformationTooltip";
import { useInput } from "./useInput";
import { EyeIcon, EyeOffIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

type InputElement = HTMLInputElement | HTMLTextAreaElement;

export interface TextFieldProps extends Omit<InputProps, "size" | "onKeyDown"> {
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
  // Widened over InputProps, which only describes the <input> branch, so the
  // handler is callable with either element's event and needs no cast.
  onKeyDown?: React.KeyboardEventHandler<InputElement>;
  // Textarea-specific props
  rows?: number;
  amountPrefix?: string;
  amountSuffix?: string;
}

export const Input = forwardRef<InputElement, TextFieldProps>(function Input(
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
    "rounded-xl border border-zinc-200 bg-white px-4 shadow-none w-full",
    "font-normal text-black",
    "placeholder:font-normal placeholder:text-zinc-500",
    // Focus and hover states
    "focus:border-purple-400 focus:shadow-none focus:outline-none focus:ring-1 focus:ring-purple-400 focus:ring-offset-0",
    className,
  );

  const errorStyles =
    error && "!border !border-red-500 focus:border-red-500 focus:ring-red-500";

  const renderInput = () => {
    if (props.type === "textarea") {
      return (
        <textarea
          ref={ref as React.Ref<HTMLTextAreaElement>}
          className={cn(
            baseStyles,
            errorStyles,
            "-mb-1 h-auto min-h-[2.875rem]",
            // Size variants for textarea
            size === "small" && [
              "min-h-[2.25rem]", // 36px minimum
              "py-2",
              "text-sm leading-[22px]",
              "placeholder:text-sm placeholder:leading-[22px]",
            ],
            size === "medium" && [
              "min-h-[2.875rem] text-sm leading-[22px]", // 46px minimum (current default)
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
          className={cn(
            baseStyles,
            errorStyles,
            // Size variants
            size === "small" && [
              "h-[2.25rem]",
              "py-2",
              "text-sm leading-[22px]",
              "placeholder:text-sm placeholder:leading-[22px]",
            ],
            size === "medium" && ["h-[2.875rem]", "py-2.5"],
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
        className={cn(
          baseStyles,
          errorStyles,
          // Add padding for password toggle button
          isPasswordType && "pr-12",
          // Size variants
          size === "small" && [
            "h-[2.25rem]", // 36px
            "py-2",
            "text-sm leading-[22px]", // 14px font, 22px line height
            "placeholder:text-sm placeholder:leading-[22px]",
          ],
          size === "medium" && [
            "h-[2.875rem]", // 46px (current default)
            "py-2.5",
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
      onBlur={isPasswordType ? handleWrapperBlur : undefined}
      className={cn("relative w-full", wrapperClassName)}
    >
      {renderInput()}
      {isPasswordType && (
        <button
          type="button"
          onClick={handleTogglePassword}
          disabled={props.disabled}
          className="absolute right-4 top-1/2 -translate-y-1/2 text-zinc-400 transition-colors hover:text-zinc-600 disabled:cursor-not-allowed disabled:opacity-50"
          aria-label={showPassword ? "Hide password" : "Show password"}
          aria-pressed={showPassword}
          aria-controls={inputId}
        >
          {showPassword ? (
            <Icon icon={EyeIcon} size={16} />
          ) : (
            <Icon icon={EyeOffIcon} size={16} />
          )}
        </button>
      )}
    </div>
  );

  const inputWithError = (
    <div className={cn("relative mb-6 w-full", wrapperClassName)}>
      {input}
      <Text
        variant="small-medium"
        as="span"
        className={cn(
          "absolute left-0 top-full mt-1 !text-red-500 transition-opacity duration-200",
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
    <div className="flex w-full flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1">
          <label htmlFor={inputId}>
            <Text
              variant={labelVariant}
              as="span"
              className={cn("text-black", labelClassName)}
            >
              {label}
            </Text>
          </label>
          {labelTooltip ? (
            <InformationTooltip description={labelTooltip} iconSize={20} />
          ) : null}
        </div>
        {hint ? (
          <Text variant="small" as="span" className="!text-zinc-400">
            {hint}
          </Text>
        ) : null}
      </div>
      {inputWithError}
    </div>
  );
});
