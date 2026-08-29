import { Button } from "@/components/atoms/Button/Button";
import {
  DropdownMenu,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import * as ContextMenu from "@radix-ui/react-context-menu";
import type { Meta, StoryObj } from "@storybook/nextjs";
import {
  SecondaryDropdownMenuContent,
  SecondaryDropdownMenuItem,
  SecondaryDropdownMenuSeparator,
  SecondaryMenuContent,
  SecondaryMenuItem,
  SecondaryMenuSeparator,
} from "./SecondaryMenu";
import {
  Copy01Icon,
  Delete02Icon,
  LinkSquare01Icon,
  MoreVerticalIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

const meta: Meta = {
  title: "Molecules/SecondaryMenu",
  component: SecondaryMenuContent,
};

export default meta;
type Story = StoryObj<typeof SecondaryMenuContent>;

export const ContextMenuExample: Story = {
  render: () => (
    <div className="flex h-96 items-center justify-center">
      <ContextMenu.Root>
        <ContextMenu.Trigger asChild>
          <div className="flex h-32 w-64 cursor-pointer items-center justify-center rounded-lg border border-gray-300 bg-gray-50 dark:border-gray-600 dark:bg-gray-800">
            Right-click me
          </div>
        </ContextMenu.Trigger>
        <SecondaryMenuContent>
          <SecondaryMenuItem onSelect={() => alert("Copy")}>
            <Icon
              icon={Copy01Icon}
              size={20}
              className="mr-2 dark:text-gray-100"
            />
            <span className="dark:text-gray-100">Copy</span>
          </SecondaryMenuItem>
          <SecondaryMenuItem onSelect={() => alert("Open agent")}>
            <Icon
              icon={LinkSquare01Icon}
              size={20}
              className="mr-2 dark:text-gray-100"
            />
            <span className="dark:text-gray-100">Open agent</span>
          </SecondaryMenuItem>
          <SecondaryMenuSeparator />
          <SecondaryMenuItem
            variant="destructive"
            onSelect={() => alert("Delete")}
          >
            <Icon
              icon={Delete02Icon}
              size={20}
              className="mr-2 text-red-500 dark:text-red-400"
            />
            <span className="dark:text-red-400">Delete</span>
          </SecondaryMenuItem>
        </SecondaryMenuContent>
      </ContextMenu.Root>
    </div>
  ),
};

export const DropdownMenuExample: Story = {
  render: () => (
    <div className="flex h-96 items-center justify-center">
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="secondary" size="small">
            <Icon icon={MoreVerticalIcon} size={16} />
          </Button>
        </DropdownMenuTrigger>
        <SecondaryDropdownMenuContent side="right" align="start">
          <SecondaryDropdownMenuItem onClick={() => alert("Copy")}>
            <Icon
              icon={Copy01Icon}
              size={20}
              className="mr-2 dark:text-gray-100"
            />
            <span className="dark:text-gray-100">Copy</span>
          </SecondaryDropdownMenuItem>
          <SecondaryDropdownMenuItem onClick={() => alert("Open agent")}>
            <Icon
              icon={LinkSquare01Icon}
              size={20}
              className="mr-2 dark:text-gray-100"
            />
            <span className="dark:text-gray-100">Open agent</span>
          </SecondaryDropdownMenuItem>
          <SecondaryDropdownMenuSeparator />
          <SecondaryDropdownMenuItem
            variant="destructive"
            onClick={() => alert("Delete")}
          >
            <Icon
              icon={Delete02Icon}
              size={20}
              className="mr-2 text-red-500 dark:text-red-400"
            />
            <span className="dark:text-red-400">Delete</span>
          </SecondaryDropdownMenuItem>
        </SecondaryDropdownMenuContent>
      </DropdownMenu>
    </div>
  ),
};
