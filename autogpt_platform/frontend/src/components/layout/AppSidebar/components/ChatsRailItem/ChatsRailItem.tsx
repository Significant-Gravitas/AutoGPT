import { Icon } from "@/components/atoms/Icon/Icon";
import {
  SidebarGroup,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/molecules/Sidebar/Sidebar";
import { BubbleChatIcon } from "@hugeicons/core-free-icons";

interface Props {
  onClick: () => void;
}

export function ChatsRailItem({ onClick }: Props) {
  return (
    <SidebarGroup spacing="tight">
      <SidebarMenu>
        <SidebarMenuItem>
          <SidebarMenuButton tooltip="Chats" onClick={onClick} variant="nav">
            <Icon
              icon={BubbleChatIcon}
              tone="emphasis-soft"
              className="size-4 group-data-[collapsible=icon]:size-4.5"
            />
            <span className="truncate">Chats</span>
          </SidebarMenuButton>
        </SidebarMenuItem>
      </SidebarMenu>
    </SidebarGroup>
  );
}
