import * as React from "react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Separator } from "@/components/ui/separator";
import {
  Popover,
  PopoverTrigger,
  PopoverContent,
} from "@/components/ui/popover";
import {
  IconType,
  IconEdit,
  IconLayoutDashboard,
  IconUploadCloud,
  IconSettings,
  IconLogOut,
  IconRefresh,
  IconMarketplace,
  IconLibrary,
  IconBuilder,
} from "../ui/icons";
import Link from "next/link";

interface ProfilePopoutMenuProps {
  userName?: string;
  userEmail?: string;
  avatarSrc?: string;
  hideNavBarUsername?: boolean;
  menuItemGroups: {
    groupName?: string;
    items: {
      icon: IconType;
      text: string;
      href?: string;
      onClick?: () => void;
    }[];
  }[];
}

interface PopoutMenuItemProps {
  icon: IconType;
  text: React.ReactNode;
  href?: string;
  onClick?: () => void;
}

const PopoutMenuItem: React.FC<PopoutMenuItemProps> = ({
  icon,
  text,
  href,
  onClick,
}) => {
  const getIcon = (iconType: IconType) => {
    let iconClass = "w-6 h-6 relative";
    const getIconWithAccessibility = (
      Icon: React.ComponentType<any>,
      label: string,
    ) => (
      <Icon className={iconClass} role="img" aria-label={label}>
        <title>{label}</title>
      </Icon>
    );

    switch (iconType) {
      case IconType.Marketplace:
        return getIconWithAccessibility(IconMarketplace, "Marketplace");
      case IconType.Library:
        return getIconWithAccessibility(IconLibrary, "Library");
      case IconType.Builder:
        return getIconWithAccessibility(IconBuilder, "Builder");
      case IconType.Edit:
        return getIconWithAccessibility(IconEdit, "Edit");
      case IconType.LayoutDashboard:
        return getIconWithAccessibility(IconLayoutDashboard, "Dashboard");
      case IconType.UploadCloud:
        return getIconWithAccessibility(IconUploadCloud, "Upload");
      case IconType.Settings:
        return getIconWithAccessibility(IconSettings, "Settings");
      case IconType.LogOut:
        return getIconWithAccessibility(IconLogOut, "Log Out");
      default:
        return getIconWithAccessibility(IconRefresh, "Refresh");
    }
  };
  if (onClick && href) {
    console.warn("onClick and href are both defined");
  }
  const content = (
    <div className="inline-flex w-full items-center justify-start gap-2.5 hover:rounded hover:bg-[#e0e0e0]">
      {getIcon(icon)}
      <div className="font-['Inter'] text-base font-normal leading-7 text-[#474747]">
        {text}
      </div>
    </div>
  );

  if (onClick) {
    return (
      <div className="w-full" onClick={onClick}>
        {content}
      </div>
    );
  }

  if (href) {
    return (
      <Link href={href} className="w-full">
        {content}
      </Link>
    );
  }

  return content;
};

export const ProfilePopoutMenu: React.FC<ProfilePopoutMenuProps> = ({
  userName,
  userEmail,
  avatarSrc,
  menuItemGroups,
}) => {
  const popupId = React.useId();

  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          className="flex cursor-pointer items-center space-x-3"
          aria-label="Open profile menu"
          aria-controls={popupId}
          aria-haspopup="true"
        >
          <Avatar className="h-10 w-10">
            <AvatarImage src={avatarSrc} alt="" aria-hidden="true" />
            <AvatarFallback aria-hidden="true">
              {userName?.charAt(0) || "U"}
            </AvatarFallback>
          </Avatar>
        </button>
      </PopoverTrigger>

      <PopoverContent
        id={popupId}
        className="flex h-[380px] w-[300px] flex-col items-start justify-start gap-4 rounded-[26px] bg-zinc-400/70 p-6 shadow backdrop-blur-2xl"
      >
        {/* Header with avatar and user info */}
        <div className="inline-flex items-center justify-start gap-4 self-stretch">
          <Avatar className="h-[60px] w-[60px]">
            <AvatarImage src={avatarSrc} alt="" aria-hidden="true" />
            <AvatarFallback aria-hidden="true">
              {userName?.charAt(0) || "U"}
            </AvatarFallback>
          </Avatar>
          <div className="relative h-[47px] w-[173px]">
            <div className="absolute left-0 top-0 font-['Geist'] text-base font-semibold leading-7 text-white">
              {userName}
            </div>
            <div className="absolute left-0 top-[23px] font-['Geist'] text-base font-normal leading-normal text-white">
              {userEmail}
            </div>
          </div>
        </div>

        {/* Menu items */}
        <div className="flex w-full flex-col items-start justify-start gap-1.5 rounded-[23px]">
          {menuItemGroups.map((group, groupIndex) => (
            <div
              key={groupIndex}
              className="flex w-full flex-col items-start justify-start gap-5 rounded-[18px] bg-white p-3.5"
            >
              {group.items.map((item, itemIndex) => (
                <div
                  key={itemIndex}
                  className="inline-flex w-full items-center justify-start gap-2.5"
                  onClick={item.onClick}
                  role="button"
                  tabIndex={0}
                >
                  <div className="relative h-6 w-6">{getIcon(item.icon)}</div>
                  <div className="font-['Geist'] text-base font-medium leading-normal text-neutral-800">
                    {item.text}
                  </div>
                </div>
              ))}
            </div>
          ))}
        </div>
      </PopoverContent>
    </Popover>
  );
};

// Helper function to get the icon component
const getIcon = (icon: IconType) => {
  const iconClass = "w-6 h-6";
  switch (icon) {
    case IconType.LayoutDashboard:
      return <IconLayoutDashboard className={iconClass} />;
    case IconType.UploadCloud:
      return <IconUploadCloud className={iconClass} />;
    case IconType.Edit:
      return <IconEdit className={iconClass} />;
    case IconType.Settings:
      return <IconSettings className={iconClass} />;
    case IconType.LogOut:
      return <IconLogOut className={iconClass} />;
    default:
      return null;
  }
};
