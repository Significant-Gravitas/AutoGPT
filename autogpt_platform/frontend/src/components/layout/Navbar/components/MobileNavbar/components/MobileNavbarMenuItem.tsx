import { IconType } from "@/components/ui/icons";
import { cn } from "@/lib/utils";
import Link from "next/link";
import { getAccountMenuOptionIcon } from "../../../helpers";

interface Props {
  icon: IconType;
  isActive: boolean;
  text: string;
  href?: string;
  onClick?: () => void;
}

export function MobileNavbarMenuItem({
  icon,
  isActive,
  text,
  href,
  onClick,
}: Props) {
  const content = (
    <div className="inline-flex w-full items-center justify-start gap-4 hover:rounded hover:bg-[#e0e0e0] dark:hover:bg-[#3a3a3a]">
      {getAccountMenuOptionIcon(icon)}
      <div className="relative">
        <div
          className={cn(
            "font-sans text-base font-normal leading-7",
            isActive
              ? "font-semibold text-[#272727] dark:text-[#ffffff]"
              : "text-[#474747] dark:text-[#cfcfcf]",
          )}
        >
          {text}
        </div>
        {isActive && (
          <div className="absolute bottom-[-4px] left-0 h-[2px] w-full bg-[#272727] dark:bg-[#ffffff]"></div>
        )}
      </div>
    </div>
  );

  if (onClick)
    return (
      <div className="w-full" onClick={onClick}>
        {content}
      </div>
    );
  if (href)
    return (
      <Link href={href} className="w-full">
        {content}
      </Link>
    );
  return content;
}
