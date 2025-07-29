import { Input as BaseInput } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { Eye, EyeSlash } from "@phosphor-icons/react";
import { ReactNode, useState } from "react";
import { Text } from "../Text/Text";
import { useInput } from "./useInput";

interface BaseFieldProps {
  label: string;
  id: string;
  hideLabel?: boolean;
  error?: string;
  hint?: ReactNode;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
  value?: string;
  onChange?: React.ChangeEventHandler<HTMLInputElement>;
}

export interface TextFieldProps extends BaseFieldProps {
  decimalCount?: number; // Only used for type="amount"
  type?:
    | "text"
    | "email"
    | "password"
    | "number"
    | "amount"
    | "tel"
    | "url"
    | "textarea";
  // Textarea-specific props
  rows?: number;
}

export function Input({
  className,
  label,
  placeholder,
  hideLabel = false,
  decimalCount,
  hint,
  error,
  type = "text",
  rows = 3,
  id,
  disabled,
  value,
  onChange,
  ...rest
}: TextFieldProps) {
  const { handleInputChange, handleTextareaChange } = useInput({
    type,
    decimalCount,
    onChange,
  });
  const [showPassword, setShowPassword] = useState(false);

  const isPasswordType = type === "password";
  const inputType = showPassword ? "text" : type;

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
    // Override the default input styles with Figma design
    "h-[2.875rem] rounded-3xl border border-zinc-200 bg-white px-4 py-2.5 shadow-none",
    "font-normal text-black text-sm",
    "placeholder:font-normal placeholder:text-zinc-400",
    // Focus and hover states
    "focus:border-zinc-400 focus:shadow-none focus:outline-none focus:ring-1 focus:ring-zinc-400 focus:ring-offset-0",
    // Error state
    error &&
      "border-1.5 border-red-500 focus:border-red-500 focus:ring-red-500",
    className,
  );

  const renderInput = () => {
    if (type === "textarea") {
      return (
        <textarea
          className={cn(baseStyles, "h-auto min-h-[2.875rem] w-full py-2.5")}
          placeholder={placeholder || label}
          onChange={handleTextareaChange}
          rows={rows}
          {...(hideLabel ? { "aria-label": label } : {})}
          id={id}
          disabled={disabled}
          value={value}
        />
      );
    }

    return (
      <BaseInput
        className={cn(
          baseStyles,
          // Add padding for password toggle button
          isPasswordType && "pr-12",
        )}
        placeholder={placeholder || label}
        onChange={handleInputChange}
        {...(hideLabel ? { "aria-label": label } : {})}
        id={id}
        disabled={disabled}
        value={value}
        type={inputType}
        {...rest}
      />
    );
  };

  const input = (
    <div className="relative">
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
    <div className="relative mb-6">
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
    <label htmlFor={id} className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <Text variant="body-medium" as="span" className="text-black">
          {label}
        </Text>
        {hint}
      </div>
      {inputWithError}
    </label>
  );
}
