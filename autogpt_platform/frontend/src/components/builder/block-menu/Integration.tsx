import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Plus } from "lucide-react";
import Image from "next/image";
import React, { ButtonHTMLAttributes } from "react";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  title?: string;
  description?: string;
  icon_url?: string;
  number_of_blocks?: number;
}

const Integration: React.FC<Props> = ({
  title,
  icon_url,
  description,
  className,
  number_of_blocks,
  ...rest
}) => {
  return (
    <Button
      className={cn(
        "group flex h-16 w-full min-w-[7.5rem] items-center justify-start space-x-3 whitespace-normal rounded-[0.75rem] bg-zinc-50 px-[0.875rem] py-[0.625rem] text-start shadow-none hover:bg-zinc-100 focus:ring-0 active:border active:border-zinc-300 active:bg-zinc-50 disabled:pointer-events-none",
      )}
      {...rest}
    >
      <div className="relative h-[2.625rem] w-[2.625rem] rounded-[0.5rem] bg-white">
        {icon_url && (
          <Image
            src={icon_url}
            alt="integration-icon"
            fill
            className="w-full object-contain group-disabled:opacity-50"
          />
        )}
      </div>

      <div className="w-full">
        <div className="flex items-center justify-between gap-2">
          <p className="line-clamp-1 flex-1 font-sans text-sm font-medium leading-[1.375rem] text-zinc-700 group-disabled:text-zinc-400">
            {title}
          </p>
          <span className="flex h-[1.375rem] w-[1.6875rem] items-center justify-center rounded-[1.25rem] bg-[#f0f0f0] p-1.5 font-sans text-sm leading-[1.375rem] text-zinc-500 group-disabled:text-zinc-400">
            {number_of_blocks}
          </span>
        </div>
        <span className="line-clamp-1 font-sans text-xs font-normal leading-5 text-zinc-500 group-disabled:text-zinc-400">
          {description}
        </span>
      </div>
    </Button>
  );
};

export default Integration;
