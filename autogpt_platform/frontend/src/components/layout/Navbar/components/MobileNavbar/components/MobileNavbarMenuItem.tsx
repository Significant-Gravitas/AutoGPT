import { cn } from "@/lib/utils";
import Link from "next/link";
import { getAccountMenuOptionIcon, IconType } from "../../../helpers";

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
    <div className="inline-flex w-full items-center justify-start gap-4 py-2 hover:rounded-sm hover:bg-zinc-200">
      {getAccountMenuOptionIcon(icon)}
      <div className="relative">
        <div
          className={cn(
            "font-sans text-base leading-7 font-normal",
            isActive ? "font-semibold text-zinc-900" : "text-zinc-800",
          )}
        >
          {text}
        </div>
        {isActive && (
          <div className="absolute -bottom-1 left-0 h-0.5 w-full bg-zinc-900"></div>
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
