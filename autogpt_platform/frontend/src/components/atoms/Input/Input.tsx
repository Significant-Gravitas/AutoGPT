import {
  Input as BaseInput,
  type InputProps,
} from "@/components/__legacy__/ui/input";
import { cn } from "@/lib/utils";
import { Eye, EyeSlash } from "@phosphor-icons/react";
import { ReactNode, useState } from "react";
import CurrencyInput from "react-currency-input-field";
import { Text } from "../Text/Text";
import { useInput } from "./useInput";

export interface TextFieldProps extends Omit<InputProps, "size"> {
  label: string;
  id: string;
  hideLabel?: boolean;
  decimalCount?: number; // Only used for type="amount"
  error?: string;
  hint?: ReactNode;
  size?: "small" | "medium";
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
  // Textarea-specific props
  rows?: number;
  amountPrefix?: string;
  amountSuffix?: string;
}

export function Input({
  className,
  label,
  placeholder,
  hideLabel = false,
  decimalCount,
  hint,
  error,
  size = "medium",
  wrapperClassName,
  amountPrefix,
  amountSuffix,
  ...props
}: TextFieldProps) {
  const { handleInputChange, handleTextareaChange, handleAmountValueChange } =
    useInput({
      type: props.type,
      onChange: props.onChange,
      decimalCount,
    });
  const [showPassword, setShowPassword] = useState(false);

  const isPasswordType = props.type === "password";
  const inputType = showPassword ? "text" : props.type;

  function handleMouseDown() {
    setShowPassword(true);
  }

  function handleMouseUp() {
    setShowPassword(false);
  }

  function handleMouseLeave() {
    setShowPassword(false);
  }

  const baseStyles = cn(
    // Base styles
    "rounded-3xl border border-zinc-200 bg-white px-4 shadow-none w-full",
    "font-normal text-black",
    "placeholder:font-normal placeholder:text-zinc-400",
    // Focus and hover states
    "focus:border-zinc-400 focus:shadow-none focus:outline-none focus:ring-1 focus:ring-zinc-400 focus:ring-offset-0",
    className,
  );

  const errorStyles =
    error && "!border !border-red-500 focus:border-red-500 focus:ring-red-500";

  const renderInput = () => {
    if (props.type === "textarea") {
      return (
        <textarea
          className={cn(
            baseStyles,
            errorStyles,
            "-mb-1 h-auto min-h-[2.875rem] rounded-medium",
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
          rows={props.rows || 3}
          {...(hideLabel ? { "aria-label": label } : {})}
          id={props.id}
          disabled={props.disabled}
          value={props.value}
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
          id={props.id}
          name={props.name}
          disabled={props.disabled}
          inputMode="decimal"
          decimalsLimit={decimalCount ?? 4}
          allowDecimals={decimalCount !== 0}
          groupSeparator=","
          decimalSeparator="."
          allowNegativeValue
          {...(hideLabel ? { "aria-label": label } : {})}
          // Pass through common handlers
          onBlur={props.onBlur as any}
          onFocus={props.onFocus as any}
          prefix={amountPrefix}
          suffix={amountSuffix}
        />
      );
    }

    return (
      <BaseInput
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
        {...(hideLabel ? { "aria-label": label } : {})}
        {...props}
        type={inputType}
      />
    );
  };

  const input = (
    <div className={cn("relative w-full", wrapperClassName)}>
      {renderInput()}
      {isPasswordType && (
        <button
          type="button"
          onMouseDown={handleMouseDown}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
          className="absolute right-4 top-1/2 -translate-y-1/2 text-zinc-400 transition-colors hover:text-zinc-600"
          aria-label="Press and hold to show password"
        >
          {showPassword ? <Eye size={16} /> : <EyeSlash size={16} />}
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
    <label htmlFor={props.id} className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <Text variant="large-medium" as="span" className="text-black">
          {label}
        </Text>
        {hint ? (
          <Text variant="small" as="span" className="!text-zinc-400">
            {hint}
          </Text>
        ) : null}
      </div>
      {inputWithError}
    </label>
  );
}
