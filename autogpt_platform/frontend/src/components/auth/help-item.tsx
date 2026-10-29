import { ExternalLink } from "lucide-react";
import Link from "next/link";

interface HelpItemProps {
  title: string;
  description?: string;
  linkText?: string;
  href?: string;
}

export function HelpItem({
  title,
  description,
  linkText,
  href = "",
}: HelpItemProps) {
  const external = !href.startsWith("/");

  return (
    <div className="p-4">
      <h3 className="mb-1 text-sm font-medium text-foreground-base">{title}</h3>
      <p className="text-sm text-foreground-secondary">
        {description}{" "}
        {linkText && (
          <Link
            href={href}
            className="inline-flex items-center font-medium text-foreground-base hover:text-foreground-body"
          >
            {linkText}
            {external && <ExternalLink className="ml-1 h-3 w-3" />}
          </Link>
        )}
      </p>
    </div>
  );
}
