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
      <h3
        // TODO(tokens): no exact role for text-slate-950 (#020617)
        className="mb-1 text-sm font-medium text-slate-950"
      >
        {title}
      </h3>
      <p
        // TODO(tokens): no exact role for text-slate-600 (#5B6A7E)
        className="text-sm text-slate-600"
      >
        {description}{" "}
        {linkText && (
          <Link
            href={href}
            // TODO(tokens): no exact role for text-slate-950 (#020617), hover:text-slate-700 (#475263)
            className="inline-flex items-center font-medium text-slate-950 hover:text-slate-700"
          >
            {linkText}
            {external && <ExternalLink className="ml-1 h-3 w-3" />}
          </Link>
        )}
      </p>
    </div>
  );
}
