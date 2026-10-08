"use client";

import {
  DropdownMenu as KobraDropdownMenu,
  DropdownMenuCheckboxItem as KobraDropdownMenuCheckboxItem,
  DropdownMenuContent as KobraDropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem as KobraDropdownMenuItem,
  DropdownMenuPortal,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem as KobraDropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuShortcut,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger as KobraDropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";
import * as React from "react";

type AsChildProps<P> = P & {
  /** Radix form: the single child becomes the rendered element. */
  asChild?: boolean;
};

function renderFromChild<P extends { children?: React.ReactNode }>({
  asChild,
  children,
  ...props
}: AsChildProps<P>) {
  if (asChild && React.isValidElement(children)) {
    return { ...props, render: children };
  }
  return { ...props, children };
}

function DropdownMenuTrigger(
  props: AsChildProps<React.ComponentProps<typeof KobraDropdownMenuTrigger>>,
) {
  return <KobraDropdownMenuTrigger {...renderFromChild(props)} />;
}

function DropdownMenuContent({
  align = "center",
  ...props
}: React.ComponentProps<typeof KobraDropdownMenuContent>) {
  return <KobraDropdownMenuContent align={align} {...props} />;
}

// Base UI's group label must sit inside a Group; the house label was
// free-standing (Radix), so it stays a plain heading row.
function DropdownMenuLabel({
  className,
  inset,
  ...props
}: React.ComponentProps<"div"> & { inset?: boolean }) {
  return (
    <div
      data-slot="dropdown-menu-label"
      className={cn(
        "px-2.5 py-1 text-xs font-medium text-muted-foreground",
        inset && "ps-8.5",
        className,
      )}
      {...props}
    />
  );
}

type ItemProps = AsChildProps<
  Omit<React.ComponentProps<typeof KobraDropdownMenuItem>, "onSelect">
> & {
  /** Radix name for the activation handler; Base UI calls it `onClick`.
   *  `preventDefault()` keeps the menu open, as it did there. */
  onSelect?: (event: React.MouseEvent<HTMLDivElement>) => void;
};

function DropdownMenuItem({ onSelect, onClick, ...props }: ItemProps) {
  return (
    <KobraDropdownMenuItem
      onClick={(event) => {
        onClick?.(event);
        if (!onSelect) return;
        onSelect(event);
        if (event.defaultPrevented) event.preventBaseUIHandler();
      }}
      {...renderFromChild(props)}
    />
  );
}

type MenuActionsRef = NonNullable<
  React.ComponentProps<typeof KobraDropdownMenu>["actionsRef"]
>;

const MenuActionsContext = React.createContext<MenuActionsRef | null>(null);

function DropdownMenu({
  actionsRef,
  ...props
}: React.ComponentProps<typeof KobraDropdownMenu>) {
  const ownActionsRef: MenuActionsRef = React.useRef(null);
  const ref = actionsRef ?? ownActionsRef;
  return (
    <MenuActionsContext value={ref}>
      <KobraDropdownMenu actionsRef={ref} {...props} />
    </MenuActionsContext>
  );
}

interface PickProps {
  /** Radix closed the menu on a radio or checkbox pick; Base UI does not.
   *  Defaults to `true` to keep the Radix behaviour. */
  closeOnClick?: boolean;
  /** Radix name for the pick handler; `preventDefault()` keeps the menu
   *  open, as it did there. */
  onSelect?: (event: React.MouseEvent<HTMLDivElement>) => void;
  onClick?: React.MouseEventHandler<HTMLDivElement>;
}

function usePickProps({ closeOnClick = true, onSelect, onClick }: PickProps) {
  const actionsRef = React.useContext(MenuActionsContext);
  if (!onSelect) return { closeOnClick, onClick };
  return {
    closeOnClick: false,
    onClick(event: React.MouseEvent<HTMLDivElement>) {
      onClick?.(event);
      onSelect(event);
      if (closeOnClick && !event.defaultPrevented) actionsRef?.current?.close();
    },
  };
}

type RadioItemProps = Omit<
  React.ComponentProps<typeof KobraDropdownMenuRadioItem>,
  keyof PickProps
> &
  PickProps;

function DropdownMenuRadioItem({
  closeOnClick,
  onSelect,
  onClick,
  ...props
}: RadioItemProps) {
  const pick = usePickProps({ closeOnClick, onSelect, onClick });
  return <KobraDropdownMenuRadioItem {...props} {...pick} />;
}

type CheckboxItemProps = Omit<
  React.ComponentProps<typeof KobraDropdownMenuCheckboxItem>,
  keyof PickProps
> &
  PickProps;

function DropdownMenuCheckboxItem({
  closeOnClick,
  onSelect,
  onClick,
  ...props
}: CheckboxItemProps) {
  const pick = usePickProps({ closeOnClick, onSelect, onClick });
  return <KobraDropdownMenuCheckboxItem {...props} {...pick} />;
}

export {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuCheckboxItem,
  DropdownMenuRadioItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuShortcut,
  DropdownMenuGroup,
  DropdownMenuPortal,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuRadioGroup,
};
