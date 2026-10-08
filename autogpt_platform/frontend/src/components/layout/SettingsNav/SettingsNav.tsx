"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Link } from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";
import { Sheet } from "@/components/molecules/Sheet/Sheet";
import { Menu01Icon } from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { cn } from "@/lib/utils";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

export interface SettingsNavLink {
  label: string;
  href: string;
  icon: IconSvgElement;
}

interface Props {
  /** Accessible name of the navigation, also the title of the mobile sheet. */
  label: string;
  links: SettingsNavLink[];
  className?: string;
}

function isActive(pathname: string, href: string, links: SettingsNavLink[]) {
  if (pathname === href) return true;
  // "/profile" must not light up for "/profile/settings" when the latter has
  // its own entry; only the longest matching href is the active one.
  const longest = links
    .map((link) => link.href)
    .filter((candidate) => pathname.startsWith(`${candidate}/`))
    .sort((a, b) => b.length - a.length)[0];
  return longest === href;
}

export function SettingsNav({ label, links, className }: Props) {
  const pathname = usePathname() ?? "";
  const [open, setOpen] = useState(false);

  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  function renderList() {
    return (
      <ul className="flex flex-col gap-1">
        {links.map((link) => {
          const active = isActive(pathname, link.href, links);
          return (
            <li key={link.href} aria-current={active ? "page" : undefined}>
              <Link
                href={link.href}
                className={cn(
                  "flex w-full items-center gap-2.5 rounded-xl px-3 py-2.5 transition-colors hover:bg-muted hover:no-underline",
                  active && "bg-muted",
                )}
              >
                <Icon icon={link.icon} size={20} aria-hidden />
                <Text
                  variant="body-medium"
                  as="span"
                  tone={active ? "primary" : "secondary"}
                >
                  {link.label}
                </Text>
              </Link>
            </li>
          );
        })}
      </ul>
    );
  }

  return (
    <>
      <div className="lg:hidden">
        <Sheet
          title={label}
          side="left"
          open={open}
          onOpenChange={setOpen}
          className="w-72 sm:max-w-72"
          trigger={
            <Button
              variant="icon"
              size="icon-lg"
              aria-label={`Open ${label.toLowerCase()} menu`}
              withTooltip={false}
            >
              <Icon icon={Menu01Icon} size={20} aria-hidden />
            </Button>
          }
        >
          <nav aria-label={label}>{renderList()}</nav>
        </Sheet>
      </div>
      <nav
        aria-label={label}
        className={cn(
          "hidden w-60 shrink-0 self-start rounded-2xl border border-border bg-card p-3 lg:block",
          className,
        )}
      >
        {renderList()}
      </nav>
    </>
  );
}
