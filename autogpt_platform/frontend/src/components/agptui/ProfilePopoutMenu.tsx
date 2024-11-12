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
        className="w-[300px] h-[380px] p-6 bg-zinc-400/70 rounded-[26px] shadow backdrop-blur-2xl flex flex-col justify-start items-start gap-4"
      >
        {/* Header with avatar and user info */}
        <div className="self-stretch inline-flex justify-start items-center gap-4">
          <Avatar className="h-[60px] w-[60px]">
            <AvatarImage src={avatarSrc} alt="" aria-hidden="true" />
            <AvatarFallback aria-hidden="true">
              {userName?.charAt(0) || "U"}
            </AvatarFallback>
          </Avatar>
          <div className="w-[173px] h-[47px] relative">
            <div className="left-0 top-0 absolute text-white text-base font-semibold font-['Geist'] leading-7">
              {userName}
            </div>
            <div className="left-0 top-[23px] absolute text-white text-base font-normal font-['Geist'] leading-normal">
              {userEmail}
            </div>
          </div>
        </div>

        {/* Menu items */}
        <div className="w-full rounded-[23px] flex flex-col justify-start items-start gap-1.5">
          {menuItemGroups.map((group, groupIndex) => (
            <div
              key={groupIndex}
              className="w-full p-3.5 bg-white rounded-[18px] flex flex-col justify-start items-start gap-5"
            >
              {group.items.map((item, itemIndex) => (
                <div
                  key={itemIndex}
                  className="w-full inline-flex justify-start items-center gap-2.5"
                  onClick={item.onClick}
                  role="button"
                  tabIndex={0}
                >
                  <div className="w-6 h-6 relative">
                    {getIcon(item.icon)}
                  </div>
                  <div className="text-neutral-800 text-base font-medium font-['Geist'] leading-normal">
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

