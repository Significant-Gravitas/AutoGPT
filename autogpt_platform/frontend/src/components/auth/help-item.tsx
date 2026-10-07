import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { LinkSquare02Icon } from "@hugeicons/core-free-icons";
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
      <Text variant="body-medium" as="h3" className="mb-1 text-black">
        {title}
      </Text>
      <Text variant="body" className="text-slate-600">
        {description}{" "}
        {linkText && (
          <Link
            href={href}
            className="inline-flex items-center font-medium text-black hover:text-slate-700"
          >
            {linkText}
            {external && (
              <Icon icon={LinkSquare02Icon} className="ml-1 h-3 w-3" />
            )}
          </Link>
        )}
      </Text>
    </div>
  );
}
