import { Icon } from "@/components/atoms/Icon/Icon";
import {
  SidebarGroup,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import type { IconSvgElement } from "@hugeicons/react";
import { useNavItemClassName } from "../../useNavItemClassName";

interface Props {
  icon: IconSvgElement;
  label: string;
  isActive?: boolean;
  onClick: () => void;
}

export function NavRailItem({ icon, label, isActive = false, onClick }: Props) {
  const navItemClassName = useNavItemClassName();

  return (
    <SidebarGroup className="py-1">
      <SidebarMenu className="group-data-[collapsible=icon]:gap-1">
        <SidebarMenuItem>
          <SidebarMenuButton
            tooltip={label}
            isActive={isActive}
            onClick={onClick}
            className={navItemClassName}
          >
            <Icon
              icon={icon}
              className="size-4 text-sidebar-foreground/90 group-data-[collapsible=icon]:size-4.5"
            />
            <span className="truncate">{label}</span>
          </SidebarMenuButton>
        </SidebarMenuItem>
      </SidebarMenu>
    </SidebarGroup>
  );
}
