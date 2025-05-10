import { FC } from "react";
import Link from "next/link";
import { cn } from "@/lib/utils";
import { ExternalLink } from "lucide-react";

interface CreatorLinkProps {
  href: string;
  children: React.ReactNode;
  className?: string;
  key?: number;
}

const CreatorLink: FC<CreatorLinkProps> = ({
  href,
  children,
  className,
  key,
}) => {
  return (
    <Link
      key={key}
      href={href}
      className={cn(
        "flex h-12 w-full min-w-80 max-w-md items-center justify-between rounded-[34px] border border-neutral-600 bg-transparent px-5 py-3",
        className,
      )}
    >
      <p className="font-sans text-base font-medium text-neutral-800">
        {children}
      </p>
      <ExternalLink className="h-5 w-5 stroke-[1.5px]" />
    </Link>
  );
};

export default CreatorLink;
