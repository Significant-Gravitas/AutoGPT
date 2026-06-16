"use client";

import { Button } from "@/components/atoms/Button/Button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
} from "@/components/ui/sidebar";
import {
  CaretDownIcon,
  FlowArrowIcon,
  FolderIcon,
  type Icon,
  SparkleIcon,
  SquaresFourIcon,
  StorefrontIcon,
} from "@phosphor-icons/react";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { cn } from "@/lib/utils";
import { motion, useReducedMotion } from "framer-motion";
import Link, { useLinkStatus } from "next/link";
import { usePathname } from "next/navigation";
import { ComponentProps, ReactNode } from "react";
import { getSidebarItemVariants, sidebarContainerVariants } from "./animations";
import { AppSidebarHeader } from "./components/AppSidebarHeader/AppSidebarHeader";
import { RecentChats } from "./components/RecentChats/RecentChats";
import { useSidebarCounts } from "./useSidebarCounts";

type NavLink = {
  name: string;
  href: string;
  icon: Icon;
};

const MAIN_LINKS: NavLink[] = [
  { name: "Agents", href: "/library", icon: SquaresFourIcon },
  { name: "Marketplace", href: "/marketplace", icon: StorefrontIcon },
  { name: "Build", href: "/build", icon: FlowArrowIcon },
];

const WORKSPACE_LINKS: NavLink[] = [
  { name: "Files", href: "/artifacts", icon: FolderIcon },
];

function isLinkActive(pathname: string | null, href: string) {
  if (!pathname) return false;
  return pathname === href || pathname.startsWith(`${href}/`);
}

// Rendered inside the <Link>, so useLinkStatus reports that link's pending
// navigation — show a spinner until the destination page is reached, then
// fall back to the count badge.
function NavLinkRight({ count }: { count?: number }) {
  const { pending } = useLinkStatus();

  if (pending) {
    return (
      <LoadingSpinner size="small" className="ml-auto shrink-0 text-zinc-500" />
    );
  }

  if (count === undefined) return null;

  return (
    <span className="ml-auto shrink-0 text-sm font-normal text-zinc-500">
      {count > 100 ? "100+" : count}
    </span>
  );
}

function NavMenu({ links }: { links: NavLink[] }) {
  const pathname = usePathname();
  const counts = useSidebarCounts();

  return (
    <SidebarMenu className="gap-2">
      {links.map((link) => {
        const count = counts[link.href];
        return (
          <SidebarMenuItem key={link.href}>
            <SidebarMenuButton
              asChild
              isActive={isLinkActive(pathname, link.href)}
              className="font-medium text-zinc-700 data-[active=true]:!bg-zinc-200 hover:!bg-zinc-200"
            >
              <Link href={link.href}>
                <link.icon className="size-5" weight="bold" />
                <span className="truncate">{link.name}</span>
                <NavLinkRight count={count} />
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        );
      })}
    </SidebarMenu>
  );
}

function CollapsibleNavGroup({
  label,
  children,
  scrollable = false,
}: {
  label: string;
  children: ReactNode;
  scrollable?: boolean;
}) {
  return (
    <Collapsible
      defaultOpen
      className={cn(
        "group/collapsible",
        scrollable && "flex min-h-0 flex-1 flex-col",
      )}
    >
      <SidebarGroup
        className={cn("py-1", scrollable && "flex min-h-0 flex-1 flex-col")}
      >
        <SidebarGroupLabel asChild className="text-[13px] font-medium">
          <CollapsibleTrigger>
            {label}
            <CaretDownIcon
              weight="bold"
              className="ml-auto size-4 transition-transform group-data-[state=open]/collapsible:rotate-180"
            />
          </CollapsibleTrigger>
        </SidebarGroupLabel>
        <CollapsibleContent
          className={cn(
            "overflow-hidden data-[state=closed]:animate-collapsible-up data-[state=open]:animate-collapsible-down",
            scrollable && "flex min-h-0 flex-1 flex-col",
          )}
        >
          <SidebarGroupContent
            className={
              scrollable
                ? "mr-[10px] min-h-0 flex-1 overflow-y-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-200"
                : undefined
            }
          >
            {children}
          </SidebarGroupContent>
        </CollapsibleContent>
      </SidebarGroup>
    </Collapsible>
  );
}

type Props = ComponentProps<typeof Sidebar>;

export function AppSidebar(props: Props) {
  const reduceMotion = useReducedMotion();
  const itemVariants = getSidebarItemVariants(!!reduceMotion);

  return (
    <Sidebar {...props} className="[&_[data-sidebar=sidebar]]:bg-[#F3F3F4]">
      <AppSidebarHeader />

      <SidebarContent className="gap-2 overflow-hidden">
        <motion.div
          variants={sidebarContainerVariants}
          initial="hidden"
          animate="show"
          className="flex min-h-0 flex-1 flex-col gap-2"
        >
          <motion.div variants={itemVariants} className="px-2 pt-2">
            <Button
              as="NextLink"
              href="/copilot"
              variant="secondary"
              size="small"
              className="w-full rounded-xl bg-zinc-300 hover:bg-zinc-400"
              leftIcon={<SparkleIcon className="size-4" weight="bold" />}
            >
              New Task
            </Button>
          </motion.div>

          <motion.div variants={itemVariants}>
            <SidebarGroup className="py-1">
              <SidebarGroupContent>
                <NavMenu links={MAIN_LINKS} />
              </SidebarGroupContent>
            </SidebarGroup>
          </motion.div>

          <motion.div variants={itemVariants}>
            <CollapsibleNavGroup label="Workspace">
              <NavMenu links={WORKSPACE_LINKS} />
            </CollapsibleNavGroup>
          </motion.div>

          <motion.div
            variants={itemVariants}
            className="flex min-h-0 flex-1 flex-col"
          >
            <CollapsibleNavGroup label="Recent chats" scrollable>
              <RecentChats />
            </CollapsibleNavGroup>
          </motion.div>
        </motion.div>
      </SidebarContent>

      <SidebarRail />
    </Sidebar>
  );
}
