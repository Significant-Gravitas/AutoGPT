import { Button } from "@/components/atoms/Button/Button";
import {
  DropdownMenu,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import {
  SecondaryDropdownMenuContent,
  SecondaryDropdownMenuItem,
  SecondaryDropdownMenuSeparator,
  SecondaryMenu,
  SecondaryMenuContent,
  SecondaryMenuItem,
  SecondaryMenuSeparator,
  SecondaryMenuTrigger,
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
  parameters: { a11y: { test: "error" } },
};

export default meta;
type Story = StoryObj<typeof SecondaryMenuContent>;

export const ContextMenuExample: Story = {
  render: () => (
    <div className="flex h-96 items-center justify-center">
      <SecondaryMenu>
        <SecondaryMenuTrigger className="flex h-32 w-64 cursor-pointer items-center justify-center rounded-lg border border-border bg-muted">
          Right-click me
        </SecondaryMenuTrigger>
        <SecondaryMenuContent>
          <SecondaryMenuItem onSelect={() => alert("Copy")}>
            <Icon icon={Copy01Icon} size={20} className="mr-2" />
            <span className="">Copy</span>
          </SecondaryMenuItem>
          <SecondaryMenuItem onSelect={() => alert("Open agent")}>
            <Icon icon={LinkSquare01Icon} size={20} className="mr-2" />
            <span className="">Open agent</span>
          </SecondaryMenuItem>
          <SecondaryMenuSeparator />
          <SecondaryMenuItem
            variant="destructive"
            onSelect={() => alert("Delete")}
          >
            <Icon icon={Delete02Icon} size={20} className="mr-2 text-red-500" />
            <span className="">Delete</span>
          </SecondaryMenuItem>
        </SecondaryMenuContent>
      </SecondaryMenu>
    </div>
  ),
};

export const DropdownMenuExample: Story = {
  render: () => (
    <div className="flex h-96 items-center justify-center">
      <DropdownMenu>
        <DropdownMenuTrigger
          render={
            <Button variant="secondary" size="md" aria-label="More actions">
              <Icon icon={MoreVerticalIcon} size={16} />
            </Button>
          }
        />
        <SecondaryDropdownMenuContent side="right" align="start">
          <SecondaryDropdownMenuItem onClick={() => alert("Copy")}>
            <Icon icon={Copy01Icon} size={20} className="mr-2" />
            <span className="">Copy</span>
          </SecondaryDropdownMenuItem>
          <SecondaryDropdownMenuItem onClick={() => alert("Open agent")}>
            <Icon icon={LinkSquare01Icon} size={20} className="mr-2" />
            <span className="">Open agent</span>
          </SecondaryDropdownMenuItem>
          <SecondaryDropdownMenuSeparator />
          <SecondaryDropdownMenuItem
            variant="destructive"
            onClick={() => alert("Delete")}
          >
            <Icon icon={Delete02Icon} size={20} className="mr-2 text-red-500" />
            <span className="">Delete</span>
          </SecondaryDropdownMenuItem>
        </SecondaryDropdownMenuContent>
      </DropdownMenu>
    </div>
  ),
};
