import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Plus } from "lucide-react";
import { ButtonHTMLAttributes } from "react";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  title?: string;
  description?: string;
  ai_name?: string;
}

export const AiBlock: React.FC<Props> = ({
  title,
  description,
  className,
  ai_name,
  ...rest
}) => {
  return (
    <Button
      className={cn(
        "group flex h-[5.625rem] w-full min-w-[7.5rem] items-center justify-start space-x-3 whitespace-normal rounded-[0.75rem] bg-zinc-50 px-[0.875rem] py-[0.625rem] text-start shadow-none",
        "hover:bg-zinc-100 focus:ring-0 active:bg-zinc-100 active:ring-1 active:ring-zinc-300 disabled:pointer-events-none",
        className,
      )}
      {...rest}
    >
      <div className="flex flex-1 flex-col items-start gap-1.5">
        <div className="space-y-0.5">
          <span
            className={cn(
              "line-clamp-1 font-sans text-sm font-medium leading-[1.375rem] text-zinc-700 group-disabled:text-zinc-400",
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

        <span
          className={cn(
            "rounded-[0.75rem] bg-zinc-200 px-[0.5rem] font-sans text-xs leading-[1.25rem] text-zinc-500",
          )}
        >
          Supports {ai_name}
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