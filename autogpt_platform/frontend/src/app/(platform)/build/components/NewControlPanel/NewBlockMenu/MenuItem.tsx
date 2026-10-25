// BLOCK MENU TODO: We need to add a better hover state to it; currently it's not in the design either.

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import React, { ButtonHTMLAttributes } from "react";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  selected?: boolean;
  number?: number;
  name?: string;
  menuItemType?: string;
}

export const MenuItem: React.FC<Props> = ({
  selected = false,
  number,
  name,
  className,
  menuItemType,
  ...rest
}) => {
  return (
    <Button
      variant="legacy-default"
      data-id={menuItemType ? `menu-item-${menuItemType}` : undefined}
      className={cn(
        "flex h-[2.375rem] w-[12.875rem] justify-between whitespace-normal rounded-[0.5rem] bg-transparent p-2 pl-3 shadow-none",
        "hover:cursor-default hover:bg-zinc-100 focus:ring-0",
        selected && "bg-zinc-100",
        className,
      )}
      {...rest}
    >
      <Text
        variant="body-medium"
        as="span"
        tone="emphasis"
        className="truncate"
      >
        {name}
      </Text>
      {number !== undefined && (
        <Text variant="body" as="span" tone="secondary">
          {number > 100 ? "100+" : number}
        </Text>
      )}
    </Button>
  );
};
