import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Plus } from "lucide-react";
import Image from "next/image";
import React, { ButtonHTMLAttributes } from "react";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  title?: string;
  description?: string;
  icon_url?: string;
}

const IntegrationBlock: React.FC<Props> = ({
  title,
  icon_url,
  description,
  className,
  ...rest
}) => {
  return (
    <Button
      className={cn(
        "group flex h-16 w-full min-w-[7.5rem] items-center justify-start gap-3 whitespace-normal rounded-[0.75rem] bg-zinc-50 px-[0.875rem] py-[0.625rem] text-start shadow-none hover:bg-zinc-100 focus:ring-0 active:border active:border-zinc-300 active:bg-zinc-100 disabled:pointer-events-none",
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
      <div className="flex flex-1 flex-col items-start gap-0.5">
        <span
          className={cn(
            "line-clamp-1 font-sans text-sm font-medium leading-[1.375rem] text-zinc-800 group-disabled:text-zinc-400",
          )}
        >
          {title}
        </span>
        <span
          className={cn(
            "line-clamp-1 font-sans text-xs font-normal leading-5 text-zinc-500 group-disabled:text-zinc-400",
          )}
        >
          {description}
        </span>
      </div>
      <div
        className={cn(
          "flex h-7 w-7 items-center justify-center rounded-[0.5rem] bg-zinc-700 group-disabled:bg-zinc-400",
        )}
      >
        <Plus className="h-5 w-5 text-zinc-50" strokeWidth={2} />
      </div>
    </Button>
  );
};

export default IntegrationBlock;
