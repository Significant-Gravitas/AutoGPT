import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";
import { PlusSignIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
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
      variant="ghost"
      className={cn(
        "group flex h-[5.625rem] w-full min-w-[7.5rem] items-center justify-start gap-3 whitespace-normal rounded-xl bg-zinc-50 px-3.5 py-2.5 text-start shadow-none",
        "hover:bg-zinc-100 focus:ring-0 active:bg-zinc-100 active:ring-1 active:ring-zinc-300 disabled:pointer-events-none disabled:opacity-50",
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
            "rounded-xl bg-zinc-200 px-2 font-sans text-xs leading-5 text-zinc-500",
          )}
        >
          Supports {ai_name}
        </span>
      </div>
      <div
        className={cn(
          "flex h-7 w-7 items-center justify-center rounded-lg bg-zinc-700 group-disabled:bg-zinc-400",
        )}
      >
        <Icon icon={PlusSignIcon} className="h-5 w-5 text-zinc-50" />
      </div>
    </Button>
  );
};
