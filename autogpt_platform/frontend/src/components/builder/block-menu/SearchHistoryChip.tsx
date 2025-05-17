import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { ArrowUpRight } from "lucide-react";
import React, { ButtonHTMLAttributes } from "react";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  content?: string;
}

const SearchHistoryChip: React.FC<Props> = ({
  content,
  className,
  ...rest
}) => {
  return (
    <Button
      className={cn(
        "h-[2.25rem] space-x-1 rounded-[1.5rem] bg-zinc-50 p-[0.375rem] pr-[0.625rem] shadow-none hover:bg-zinc-100 focus:ring-0 active:border active:border-zinc-300 active:bg-zinc-100",
        className,
      )}
      {...rest}
    >
      <ArrowUpRight className="h-6 w-6 text-zinc-500" strokeWidth={1.25} />
      <span className="font-sans text-sm font-normal leading-[1.375rem] text-zinc-800">
        {content}
      </span>
    </Button>
  );
};

export default SearchHistoryChip;
