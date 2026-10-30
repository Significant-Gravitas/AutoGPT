import Link from "next/link";
import {
  Sheet,
  SheetContent,
  SheetTrigger,
} from "@/components/molecules/Dialog/components/SheetPrimitives";
import { Text } from "@/components/atoms/Text/Text";
// oxlint-disable-next-line design-system/hugeicons-only -- glyph swap is P11's
import { Menu } from "lucide-react";

export interface ProfileSidebarLink {
  text: string;
  href: string;
  icon: React.ReactNode;
}

interface Props {
  links: ProfileSidebarLink[];
}

export function ProfileSidebar({ links }: Props) {
  const renderedLinks = links.map((link, index) => (
    <Link
      key={`${link.href}-${index}`}
      href={link.href}
      className="inline-flex w-full items-center gap-2.5 rounded-xl px-3 py-3 text-foreground-emphasis hover:bg-action hover:text-action-foreground"
    >
      {link.icon}
      <Text variant="large-medium-tight" as="div" tone="inherit">
        {link.text}
      </Text>
    </Link>
  ));

  return (
    <>
      <Sheet>
        <SheetTrigger asChild>
          <button
            aria-label="Open sidebar menu"
            className="fixed top-4 left-4 z-50 flex h-14 w-14 items-center justify-center overflow-hidden rounded-lg border border-foreground-tertiary bg-surface-active px-4 py-2 font-sans text-body leading-5 font-medium tracking-tight whitespace-nowrap text-foreground-emphasis transition-colors hover:bg-surface-active/50 focus-visible:ring-1 focus-visible:ring-focus focus-visible:outline-none disabled:pointer-events-none disabled:opacity-50 md:block lg:hidden"
          >
            <Menu className="h-8 w-8 stroke-foreground-base" />
            <span className="sr-only">Open sidebar menu</span>
          </button>
        </SheetTrigger>
        <SheetContent
          side="left"
          // oxlint-disable-next-line design-system/no-restyle -- the sheet is the sidebar panel itself: no padding or border, as before
          className="z-50 w-[280px] border-none p-0 sm:w-[280px]"
        >
          <div className="h-full w-full rounded-2xl bg-surface-active">
            <div className="inline-flex h-[264px] flex-col items-start justify-start gap-6 p-3">
              {renderedLinks}
            </div>
          </div>
        </SheetContent>
      </Sheet>

      <div className="relative hidden h-[912px] w-[234px] border-none lg:block">
        <div className="h-full w-full rounded-2xl bg-surface-active">
          <div className="inline-flex h-[264px] flex-col items-start justify-start gap-6 p-3">
            {renderedLinks}
          </div>
        </div>
      </div>
    </>
  );
}
