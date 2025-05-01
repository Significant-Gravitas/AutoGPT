import React from "react";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { cn } from "@/lib/utils";

interface AutogptInputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {
  className?: string;
  label?: string;
  isDisabled?: boolean;
}

const AutogptInput: React.FC<AutogptInputProps> = ({
  label,
  isDisabled,
  ...props
}) => {
  return (
    <div className={cn("flex flex-col gap-2", isDisabled && "opacity-50")}>
      {label && (
        <Label className="font-sans text-sm font-medium leading-[1.4rem]">
          {label}
        </Label>
      )}
      <Input
        {...props}
        disabled={isDisabled}
        className={cn(
          "m-0 h-10 w-full rounded-3xl border border-zinc-300 bg-white py-2 pl-4 font-sans text-base font-normal text-zinc-800 shadow-none outline-none placeholder:text-zinc-400 focus:border-2 focus:border-[#CBD5E1] focus:shadow-none focus:ring-0",
          props.className,
        )}
      />
    </div>
  );
};

export default AutogptInput;
