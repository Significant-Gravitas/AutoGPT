import { render, screen } from "@testing-library/react";
import * as DropdownMenuPrimitive from "@radix-ui/react-dropdown-menu";
import { describe, expect, test } from "vitest";
import {
  SecondaryDropdownMenuContent,
  SecondaryDropdownMenuItem,
  SecondaryDropdownMenuSeparator,
} from "./SecondaryMenu";

describe("SecondaryMenu", () => {
  test("shows the menu action and the destructive action", () => {
    render(
      <DropdownMenuPrimitive.Root open>
        <DropdownMenuPrimitive.Trigger>Open</DropdownMenuPrimitive.Trigger>
        <SecondaryDropdownMenuContent>
          <SecondaryDropdownMenuItem>Rename</SecondaryDropdownMenuItem>
          <SecondaryDropdownMenuSeparator />
          <SecondaryDropdownMenuItem variant="destructive">
            Delete
          </SecondaryDropdownMenuItem>
        </SecondaryDropdownMenuContent>
      </DropdownMenuPrimitive.Root>,
    );

    expect(screen.getByRole("menuitem", { name: "Rename" })).toBeDefined();
    expect(screen.getByRole("menuitem", { name: "Delete" })).toBeDefined();
  });
});
