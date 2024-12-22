import { cn } from "@/lib/utils";
import Link from "next/link";

interface Props {
  className?: string,
  text: string;
  linkText?: string;
  href?: string;
}

export default function AuthBottomText({
  className = "",
  text,
  linkText,
  href = ""
}: Props) {
  return (
    <div className={cn(className, "mt-8 inline-flex w-full justify-center items-center")}>
      <span className="text-slate-950 text-sm font-medium leading-normal">{text}</span>
      {linkText && <Link href={href} className="ml-1 text-slate-950 text-sm font-medium underline leading-normal">
        {linkText}
      </Link>}
    </div>
  )
}
