import { Icon } from "@/components/atoms/Icon/Icon";
import {
  SidebarGroup,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import { BubbleChatIcon } from "@hugeicons/core-free-icons";
import { useNavItemClassName } from "../../useNavItemClassName";
import { UNMASKED_IN_REPLAY } from "@/lib/replay-privacy";
import { cn } from "@/lib/utils";

interface Props {
  onClick: () => void;
}

export function ChatsRailItem({ onClick }: Props) {
  const navItemClassName = useNavItemClassName();

  return (
    <SidebarGroup className="py-1">
      <SidebarMenu className="group-data-[collapsible=icon]:gap-1">
        <SidebarMenuItem>
          <SidebarMenuButton
            tooltip="Chats"
            onClick={onClick}
            className={navItemClassName}
          >
            <Icon
              icon={BubbleChatIcon}
              className="size-4 text-sidebar-foreground/90 group-data-[collapsible=icon]:size-4.5"
            />
            <span className={cn("truncate", UNMASKED_IN_REPLAY)}>Chats</span>
          </SidebarMenuButton>
        </SidebarMenuItem>
      </SidebarMenu>
    </SidebarGroup>
  );
}
