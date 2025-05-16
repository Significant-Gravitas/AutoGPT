import { cn } from "@/lib/utils";
import React, { ButtonHTMLAttributes } from "react";
import { LucideIcon } from "lucide-react";
import { Button } from "@/components/ui/button";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  selected?: boolean;
  icon?: LucideIcon;
}

const ControlPanel: React.FC<Props> = ({
  selected = false,
  icon: Icon,
  className,
  ...rest
}) => {
  return (
    <Button
      className={cn(
        "flex h-[4.25rem] w-[4.25rem] items-center justify-center whitespace-normal bg-white p-[1.38rem] text-zinc-800 shadow-none hover:cursor-pointer hover:bg-zinc-100 hover:text-zinc-950 focus:ring-0",
        selected &&
          "bg-violet-50 text-violet-700 hover:cursor-default hover:bg-violet-50 hover:text-violet-700",
        className,
      )}
      {...rest}
    >
      {Icon && <Icon className="h-6 w-6" strokeWidth={2} />}
    </Button>
  );
};

export default ControlPanel;
