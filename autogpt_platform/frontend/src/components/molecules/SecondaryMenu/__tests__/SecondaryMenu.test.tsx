import * as ContextMenu from "@radix-ui/react-context-menu";
import * as DropdownMenuPrimitive from "@radix-ui/react-dropdown-menu";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import {
  SecondaryDropdownMenuContent,
  SecondaryDropdownMenuItem,
  SecondaryDropdownMenuSeparator,
  SecondaryMenuContent,
  SecondaryMenuItem,
  SecondaryMenuSeparator,
} from "../SecondaryMenu";

describe("SecondaryMenu", () => {
  it("lists the dropdown items around a separator and runs the chosen one", () => {
    const onCopy = vi.fn();
    const onDelete = vi.fn();
    render(
      <DropdownMenuPrimitive.Root open>
        <DropdownMenuPrimitive.Trigger>Actions</DropdownMenuPrimitive.Trigger>
        <SecondaryDropdownMenuContent>
          <SecondaryDropdownMenuItem onSelect={onCopy}>
            Copy
          </SecondaryDropdownMenuItem>
          <SecondaryDropdownMenuSeparator />
          <SecondaryDropdownMenuItem variant="destructive" onSelect={onDelete}>
            Delete
          </SecondaryDropdownMenuItem>
        </SecondaryDropdownMenuContent>
      </DropdownMenuPrimitive.Root>,
    );

    const items = screen.getAllByRole("menuitem");
    expect(items.map((item) => item.textContent)).toEqual(["Copy", "Delete"]);
    expect(screen.getByRole("separator")).toBeDefined();

    fireEvent.click(screen.getByRole("menuitem", { name: "Delete" }));
    expect(onDelete).toHaveBeenCalledTimes(1);
    expect(onCopy).not.toHaveBeenCalled();
  });

  it("opens the context menu on right click and runs the chosen item", () => {
    const onOpen = vi.fn();
    render(
      <ContextMenu.Root>
        <ContextMenu.Trigger>Workflow card</ContextMenu.Trigger>
        <SecondaryMenuContent>
          <SecondaryMenuItem onSelect={onOpen}>Open</SecondaryMenuItem>
          <SecondaryMenuSeparator />
          <SecondaryMenuItem variant="destructive">Delete</SecondaryMenuItem>
        </SecondaryMenuContent>
      </ContextMenu.Root>,
    );

    expect(screen.queryByRole("menu")).toBeNull();
    fireEvent.contextMenu(screen.getByText("Workflow card"));

    expect(screen.getByRole("menu")).toBeDefined();
    fireEvent.click(screen.getByRole("menuitem", { name: "Open" }));
    expect(onOpen).toHaveBeenCalledTimes(1);
  });
});
