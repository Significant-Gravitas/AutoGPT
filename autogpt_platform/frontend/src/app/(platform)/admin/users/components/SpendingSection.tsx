import { Icon } from "@/components/atoms/Icon/Icon";
import { Link } from "@/components/atoms/Link/Link";
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
            <Link
              variant="secondary"
              href={link.href}
              className="flex items-center justify-between rounded-lg px-2 py-1.5 no-underline hover:bg-surface-muted"
            >
              {link.label}
              <Icon
                icon={ArrowUpRight01Icon}
                size={16}
                className="text-foreground-tertiary"
                aria-hidden
              />
            </Link>
          </li>
        ))}
      </ul>
    </AdminSection>
  );
}
