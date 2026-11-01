import { Text } from "@/components/atoms/Text/Text";
import { LinkSquare01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
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
      <Text variant="body-tight" tone="secondary">
        {description}{" "}
        {linkText && (
          <Link
            href={href}
            className="inline-flex items-center font-medium text-foreground-base hover:text-foreground-body"
          >
            {linkText}
            {external && (
              <Icon
                aria-hidden
                icon={LinkSquare01Icon}
                className="ml-1 h-3 w-3"
              />
            )}
          </Link>
        )}
      </Text>
    </div>
  );
}
