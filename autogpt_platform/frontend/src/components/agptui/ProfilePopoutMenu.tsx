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
    switch (iconType) {
      case IconType.Marketplace:
        return <IconMarketplace className={iconClass} />;
      case IconType.Library:
        return <IconLibrary className={iconClass} />;
      case IconType.Builder:
        return <IconBuilder className={iconClass} />;
      case IconType.Edit:
        return <IconEdit className={iconClass} />;
      case IconType.LayoutDashboard:
        return <IconLayoutDashboard className={iconClass} />;
      case IconType.UploadCloud:
        return <IconUploadCloud className={iconClass} />;
      case IconType.Settings:
        return <IconSettings className={iconClass} />;
      case IconType.LogOut:
        return <IconLogOut className={iconClass} />;
      default:
        return <IconRefresh className={iconClass} />;
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
  return (
    <Popover>
      <PopoverTrigger asChild>
        <div className="hidden cursor-pointer items-center space-x-5 md:flex">
          <div className="font-neue text-2xl font-medium leading-9 tracking-tight text-[#474747]">
            {userName || "Unknown User"}
          </div>
          <Avatar className="h-10 w-10">
            <AvatarImage src={avatarSrc} alt={userName || "Unknown User"} />
            <AvatarFallback>{userName?.charAt(0) || "U"}</AvatarFallback>
          </Avatar>
        </div>
      </PopoverTrigger>
      <PopoverContent className="ml-2 inline-flex w-[280px] flex-col items-start justify-start gap-3.5 rounded-[10px] border border-black/10 bg-[#efefef] px-4 py-5 shadow">
        <div className="inline-flex items-end justify-start gap-4">
          <Avatar className="h-14 w-14 border border-[#474747]">
            <AvatarImage src={avatarSrc} alt={userName || "Unknown User"} />
            <AvatarFallback>{userName?.charAt(0) || "U"}</AvatarFallback>
          </Avatar>
          <div className="relative h-14 w-[153px]">
            <div className="absolute left-0 top-0 font-['Inter'] text-lg font-semibold leading-7 text-[#474747]">
              {userName}
            </div>
            <div className="absolute left-0 top-6 font-['Inter'] text-base font-normal leading-7 text-[#474747]">
              {userEmail}
            </div>
          </div>
        </div>
        <Separator />
        {menuItemGroups.map((group, groupIndex) => (
          <React.Fragment key={groupIndex}>
            {group.items.map((item, itemIndex) => (
              <PopoutMenuItem
                key={itemIndex}
                icon={item.icon}
                text={item.text}
                onClick={item.onClick}
                href={item.href}
              />
            ))}
            {groupIndex < menuItemGroups.length - 1 && <Separator />}
          </React.Fragment>
        ))}
      </PopoverContent>
    </Popover>
  );
};
