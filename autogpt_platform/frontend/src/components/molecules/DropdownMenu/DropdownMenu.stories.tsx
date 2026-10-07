import React from "react";
import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
} from "./DropdownMenu";
import { Button } from "@/components/atoms/Button/Button";

const meta: Meta = {
  title: "Molecules/DropdownMenu",
  component: DropdownMenuContent,
  parameters: { a11y: { test: "error" } },
};

export default meta;
type Story = StoryObj<typeof DropdownMenuContent>;

export const Basic: Story = {
  render: () => (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="secondary" size="md">
          Open menu
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent>
        <DropdownMenuItem onClick={() => alert("Action 1")}>
          Action 1
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => alert("Action 2")}>
          Action 2
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={() => alert("Danger")}>
          Danger
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  ),
};
