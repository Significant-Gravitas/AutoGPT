import * as React from "react";

import { cn } from "@/lib/utils";

import { useEffect, useRef } from "react";

export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, ...props }, ref) => {
    // This ref allows the `Input` component to be both controlled and uncontrolled.
    // The HTMLvalue will only be updated if the value prop changes, but the user can still type in the input.
    const inputRef = useRef<HTMLInputElement>(null);
    const resolvedRef = (ref as any) || inputRef;

    useEffect(() => {
      if (resolvedRef.current && props.value !== undefined) {
        resolvedRef.current.value = props.value;
      }
    }, [props.value, resolvedRef]);

    return (
      <input
        type={type}
        className={cn(
          "flex h-9 w-full rounded-md border border-gray-200 bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-gray-500 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-gray-400 disabled:cursor-not-allowed disabled:opacity-50 dark:border-gray-800 dark:placeholder:text-gray-400 dark:focus-visible:ring-gray-300",
          type == "file" ? "pb-0.5 pt-1.5" : "", // fix alignment
          className,
        )}
        ref={resolvedRef}
        {...props}
      />
    );
  },
);

Input.displayName = "Input";

export { Input };
