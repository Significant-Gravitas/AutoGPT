import { cn } from "@/lib/utils";
import Link from "next/link";

interface Props {
  className?: string;
  text: string;
  linkText?: string;
  href?: string;
}

export default function AuthBottomText({
  className = "",
  text,
  linkText,
  href = "",
}: Props) {
  return (
    <div
      className={cn(
        className,
        "mt-8 inline-flex w-full items-center justify-center",
      )}
    >
      <span className="text-sm font-medium leading-normal text-slate-950">
        {text}
      </span>
      {linkText && (
        <Link
          href={href}
          className="ml-1 text-sm font-medium leading-normal text-slate-950 underline"
        >
          {linkText}
        </Link>
      )}
    </div>
  );
}
