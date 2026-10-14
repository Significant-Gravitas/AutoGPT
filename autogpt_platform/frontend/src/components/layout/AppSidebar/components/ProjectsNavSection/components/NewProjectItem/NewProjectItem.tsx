import { Icon } from "@/components/atoms/Icon/Icon";
import { SidebarMenuButton, SidebarMenuItem } from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";
import { Add01Icon } from "@hugeicons/core-free-icons";
import { useNavItemClassName } from "../../../../useNavItemClassName";
import { PROJECT_LIMIT_MESSAGE } from "../../helpers";

interface Props {
  isAtLimit: boolean;
  isCreating: boolean;
  onCreate: () => void;
}

export function NewProjectItem({ isAtLimit, isCreating, onCreate }: Props) {
  const navItemClassName = useNavItemClassName();

  // aria-disabled rather than disabled, so the button still takes hover
  // and focus and can say why it is off.
  return (
    <SidebarMenuItem>
      <SidebarMenuButton
        tooltip={
          isAtLimit
            ? { children: PROJECT_LIMIT_MESSAGE, hidden: false }
            : "New project"
        }
        aria-disabled={isAtLimit || isCreating}
        onClick={onCreate}
        className={cn(
          navItemClassName,
          "aria-disabled:pointer-events-auto aria-disabled:cursor-not-allowed aria-disabled:hover:!bg-transparent",
        )}
      >
        <Icon
          icon={Add01Icon}
          className="size-4 text-sidebar-foreground/90 group-data-[collapsible=icon]:size-4.5"
        />
        <span className="truncate">New project</span>
      </SidebarMenuButton>
    </SidebarMenuItem>
  );
}
