import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import Image from "next/image";
import React, { ButtonHTMLAttributes } from "react";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  name?: string;
  icon_url?: string;
}

const IntegrationChip: React.FC<Props> = ({
  icon_url,
  name,
  className,
  ...rest
}) => {
  return (
    <Button
      className={cn(
        "flex h-[3.25rem] w-full min-w-[7.5rem] justify-start gap-2 whitespace-normal rounded-[0.5rem] bg-zinc-50 p-2 pr-3 shadow-none hover:bg-zinc-100 focus:ring-0 active:border active:border-zinc-300 active:bg-zinc-100",
        className,
      )}
      {...rest}
    >
      <div className="relative h-9 w-9 rounded-[0.5rem] bg-transparent">
        {icon_url && (
          <Image
            src={icon_url}
            alt="integration-icon"
            fill
            className="w-full object-contain"
          />
        )}
      </div>
      <span className="font-sans text-sm font-normal leading-[1.375rem] text-zinc-800">
        {name}
      </span>
    </Button>
  );
};

export default IntegrationChip;
