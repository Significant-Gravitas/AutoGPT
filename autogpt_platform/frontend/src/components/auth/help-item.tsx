import { ExternalLink } from "lucide-react";
import Link from "next/link";

interface HelpItemProps {
  title: string;
  description: string;
  linkText: string;
  href: string;
}

export function HelpItem({
  title,
  description,
  linkText,
  href,
}: HelpItemProps) {
  return (
    <div className="p-4">
      <h3 className="mb-1 text-sm font-medium text-slate-950">{title}</h3>
      <p className="text-sm text-slate-600">
        {description}{" "}
        <Link
          href={href}
          className="inline-flex items-center font-medium text-slate-950 hover:text-slate-700"
        >
          {linkText}
          <ExternalLink className="ml-1 h-3 w-3" />
        </Link>
      </p>
    </div>
  );
}
