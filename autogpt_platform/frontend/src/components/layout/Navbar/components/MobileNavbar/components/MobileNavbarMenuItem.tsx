import { IconType } from "@/components/__legacy__/ui/icons";
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
    <div className="inline-flex w-full items-center justify-start gap-4 py-2 hover:rounded hover:bg-[#e0e0e0]">
      {getAccountMenuOptionIcon(icon)}
      <div className="relative">
        <div
          className={cn(
            "font-sans text-base font-normal leading-7",
            isActive ? "font-semibold text-[#272727]" : "text-[#474747]",
          )}
        >
          {text}
        </div>
        {isActive && (
          <div className="absolute bottom-[-4px] left-0 h-[2px] w-full bg-[#272727]"></div>
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
