import * as React from "react";

import { cn } from "@/lib/utils";

export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          "agpt-border-input agpt-shadow-input flex h-9 w-full rounded-md border bg-transparent px-3 py-1 text-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-gray-500 disabled:cursor-not-allowed disabled:opacity-50 dark:placeholder:text-gray-400",
          type == "file" ? "pb-0.5 pt-1.5" : "", // fix alignment
          className,
        )}
        ref={ref}
        {...props}
      />
    );
  },
);
Input.displayName = "Input";

const LocalValuedInput: React.FC<InputProps> = ({
  value,
  onChange,
  ...props
}) => {
  /**
   * Input component that manages its own value state.
   * This component is useful when you want to control the value of the input
   * from the parent component, but also want to allow the user to change the value.
   */
  const [inputValue, setInputValue] = React.useState(value ?? "");

  React.useEffect(() => {
    if (value !== undefined && value !== inputValue) {
      setInputValue(value);
    }
    // Note:
    // It's intended that the `inputValue` not being added to the dependency array,
    // `inputValue` should only be updated from the outside only when `value` changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
    if (onChange) onChange(e);
  };

  return <Input {...props} value={inputValue} onChange={handleChange} />;
};
LocalValuedInput.displayName = "LocalValuedInput";

export { Input, LocalValuedInput };
