import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Plus } from "lucide-react";
import Image from "next/image";
import React, { ButtonHTMLAttributes } from "react";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  title?: string;
  edited_time?: string;
  version?: number;
  image_url?: string;
}

const UGCAgentBlock: React.FC<Props> = ({
  title,
  image_url,
  edited_time,
  version,
  className,
  ...rest
}) => {
  return (
    <Button
      className={cn(
        "group flex h-[4.375rem] w-full min-w-[7.5rem] items-center justify-start gap-3 whitespace-normal rounded-[0.75rem] bg-zinc-50 p-[0.625rem] pr-[0.875rem] text-start shadow-none hover:bg-zinc-100 focus:ring-0 active:border active:border-zinc-300 active:bg-zinc-100 disabled:pointer-events-none",
      )}
      {...rest}
    >
      <div className="relative h-[3.125rem] w-[5.625rem] overflow-hidden rounded-[0.375rem] bg-white">
        {image_url && (
          <Image
            src={image_url}
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
        <div className="flex items-center space-x-2.5">
          <span
            className={cn(
              "line-clamp-1 font-sans text-xs font-normal leading-5 text-zinc-500 group-disabled:text-zinc-400",
            )}
          >
            {/* BLOCK MENU TODO: We need to create a utility to convert the edit time into relative time (e.g., "2 hours ago") */}
            Edited on {edited_time}
          </span>

          <span className="font-sans text-zinc-400">•</span>

          <span
            className={cn(
              "line-clamp-1 font-sans text-xs font-normal leading-5 text-zinc-500 group-disabled:text-zinc-400",
            )}
          >
            Version {version}
          </span>
        </div>
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

export default UGCAgentBlock;
