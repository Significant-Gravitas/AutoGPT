// BLOCK MENU TODO: We need to add a better hover state to it; currently it's not in the design either.

import { Text } from "@/components/atoms/Text/Text";
import { blockMenuButtonBase } from "./style";
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
    <button
      type="button"
      data-id={menuItemType ? `menu-item-${menuItemType}` : undefined}
      className={cn(
        blockMenuButtonBase,
        "flex h-[2.375rem] w-[12.875rem] justify-between whitespace-normal rounded-lg bg-transparent p-2 pl-3",
        "hover:cursor-default hover:bg-surface-hover",
        selected && "bg-surface-hover",
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
    </button>
  );
};
