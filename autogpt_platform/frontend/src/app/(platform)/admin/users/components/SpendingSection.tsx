import { Icon } from "@/components/atoms/Icon/Icon";
import NextLink from "next/link";
import { ArrowUpRight01Icon } from "@hugeicons/core-free-icons";
import { AdminSection } from "../../components/AdminSection";
import { getSpendingSearchHref, getUserPlatformCostsHref } from "../helpers";

interface Props {
  userId: string;
}

export function SpendingSection({ userId }: Props) {
  const links = [
    { label: "Spending history", href: getSpendingSearchHref(userId) },
    { label: "Platform cost", href: getUserPlatformCostsHref(userId) },
  ];

  return (
    <AdminSection title="Spending and cost">
      <ul className="-mx-2 flex flex-col">
        {links.map((link) => (
          <li key={link.href}>
            <NextLink
              href={link.href}
              className="flex items-center justify-between rounded-lg px-2 py-1.5 font-sans text-body font-medium text-foreground-base hover:bg-surface-muted focus:outline-none focus:ring-2 focus:ring-info focus:ring-offset-2"
            >
              {link.label}
              <span className="flex text-foreground-tertiary">
                <Icon icon={ArrowUpRight01Icon} size={16} aria-hidden />
              </span>
            </NextLink>
          </li>
        ))}
      </ul>
    </AdminSection>
  );
}
