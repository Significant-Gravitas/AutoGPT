import { Icon } from "@/components/atoms/Icon/Icon";
import { InformationTooltip } from "@/components/molecules/InformationTooltip/InformationTooltip";
import { Input as KobraInput } from "@/components/ui/input";
import { Textarea as KobraTextarea } from "@/components/ui/textarea";
import { isComposingEvent } from "@/lib/keyboard";
import { cn } from "@/lib/utils";
import { EyeIcon, EyeOffIcon } from "@hugeicons/core-free-icons";
import { forwardRef, ReactNode, useState } from "react";
import CurrencyInput from "react-currency-input-field";
import { Text } from "../Text/Text";
import type { Variant } from "../Text/helpers";
import { rowsMinHeight } from "../Textarea/helpers";
import { fieldSizeClasses, type FieldSize } from "./fieldVariants";
import { useInput } from "./useInput";

type InputElement = HTMLInputElement | HTMLTextAreaElement;

const textareaSizeClasses: Record<FieldSize, string> = {
  sm: "px-3 text-xs md:text-xs",
  md: "px-3 text-sm",
  lg: "px-4 text-sm",
};

export interface TextFieldProps extends Omit<
  React.InputHTMLAttributes<HTMLInputElement>,
  "size" | "onKeyDown"
> {
  label: string;
  id: string;
  hideLabel?: boolean;
  decimalCount?: number; // Only used for type="amount"
  error?: string;
  hint?: ReactNode;
  size?: FieldSize;
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
    size = "lg",
    labelVariant = "large-medium",
    labelClassName,
    labelTooltip,
    wrapperClassName,
    amountPrefix,
    amountSuffix,
    onKeyDown,
    rows = 3,
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

  const ariaLabel =
    props["aria-label"] ?? (hideLabel && label ? label : undefined);

  const renderInput = () => {
    if (props.type === "textarea") {
      return (
        <KobraTextarea
          ref={ref as React.Ref<HTMLTextAreaElement>}
          className={cn(
            textareaSizeClasses[size],
            "py-2 leading-snug",
            className,
          )}
          style={{ minHeight: rowsMinHeight(rows) }}
          placeholder={placeholder || label}
          onChange={handleTextareaChange}
          aria-invalid={error ? true : undefined}
          onKeyDown={guardedOnKeyDown}
          rows={rows}
          aria-label={ariaLabel}
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
          customInput={KobraInput}
          className={cn(fieldSizeClasses[size], className)}
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
          aria-invalid={error ? true : undefined}
          aria-label={ariaLabel}
          aria-labelledby={props["aria-labelledby"]}
          aria-describedby={props["aria-describedby"]}
          onBlur={props.onBlur}
          onFocus={props.onFocus}
          onKeyDown={guardedOnKeyDown}
          prefix={amountPrefix}
          suffix={amountSuffix}
        />
      );
    }

    return (
      <KobraInput
        ref={ref as React.Ref<HTMLInputElement>}
        className={cn(
          fieldSizeClasses[size],
          isPasswordType && "pr-12",
          className,
        )}
        placeholder={placeholder || label}
        onChange={handleInputChange}
        aria-invalid={error ? true : undefined}
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
          className="absolute top-1/2 right-4 -translate-y-1/2 rounded-sm text-muted-foreground focus-ring transition-colors hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
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
        tone="danger"
        className={cn(
          "absolute top-full left-0 mt-1 transition-opacity duration-200",
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
            <Text variant={labelVariant} as="span" className={labelClassName}>
              {label}
            </Text>
          </label>
          {labelTooltip ? (
            <InformationTooltip description={labelTooltip} iconSize={20} />
          ) : null}
        </div>
        {hint ? (
          <Text variant="small" as="span" tone="muted">
            {hint}
          </Text>
        ) : null}
      </div>
      {inputWithError}
    </div>
  );
});
