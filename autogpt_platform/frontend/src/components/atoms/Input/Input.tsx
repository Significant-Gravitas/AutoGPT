import { Input as BaseInput, type InputProps } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { ReactNode } from "react";
import { Text } from "../Text/Text";
import { useInput } from "./useInput";

export interface TextFieldProps extends InputProps {
  label: string;
  id: string;
  hideLabel?: boolean;
  decimalCount?: number; // Only used for type="amount"
  error?: string;
  hint?: ReactNode;
}

export function Input({
  className,
  label,
  placeholder,
  hideLabel = false,
  decimalCount,
  hint,
  error,
  ...props
}: TextFieldProps) {
  const { handleInputChange } = useInput({ ...props, decimalCount });

  const input = (
    <BaseInput
      className={cn(
        // Override the default input styles with Figma design
        "h-[2.875rem] rounded-3xl border border-zinc-200 bg-white px-4 py-2.5 shadow-none",
        "font-normal text-black",
        "placeholder:font-normal placeholder:text-zinc-400",
        // Focus and hover states
        "focus:border-zinc-400 focus:shadow-none focus:outline-none focus:ring-1 focus:ring-zinc-400 focus:ring-offset-0",
        // Error state
        error &&
          "border-1.5 border-red-500 focus:border-red-500 focus:ring-red-500",
        className,
      )}
      type={props.type}
      placeholder={placeholder || label}
      onChange={handleInputChange}
      {...(hideLabel ? { "aria-label": label } : {})}
      {...props}
    />
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
    <label htmlFor={props.id} className="flex flex-col gap-2">
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
