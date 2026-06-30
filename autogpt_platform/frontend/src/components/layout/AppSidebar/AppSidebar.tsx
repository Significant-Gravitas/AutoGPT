"use client";

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
import { ComponentProps, ReactNode, Suspense } from "react";
import { getSidebarItemVariants, sidebarContainerVariants } from "./animations";
import { AppSidebarHeader } from "./components/AppSidebarHeader/AppSidebarHeader";
import { RecentChats } from "./components/RecentChats/RecentChats";
import { SidebarSearch } from "./components/SidebarSearch/SidebarSearch";

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
// navigation — show a spinner until the destination page is reached.
function NavLinkLoader() {
  const { pending } = useLinkStatus();

  if (!pending) return null;

  return (
    <LoadingSpinner
      size="small"
      className="ml-auto !size-4 shrink-0 text-zinc-500"
    />
  );
}

// Rendered inside the New Task <Link> — swap the sparkle for a spinner while
// navigation to /copilot is pending, then back to the sparkle once it lands.
function NewTaskIcon() {
  const { pending } = useLinkStatus();

  if (pending) {
    return <LoadingSpinner size="small" className="shrink-0" />;
  }

  return <SparkleIcon className="size-4" />;
}

function NavMenu({
  links,
  leading,
}: {
  links: NavLink[];
  leading?: ReactNode;
}) {
  const pathname = usePathname();

  return (
    <SidebarMenu className="group-data-[collapsible=icon]:gap-1">
      {leading}
      {links.map((link) => (
        <SidebarMenuItem key={link.href}>
          <SidebarMenuButton
            asChild
            tooltip={link.name}
            isActive={isLinkActive(pathname, link.href)}
            className="font-normal data-[active=true]:!bg-zinc-200 data-[active=true]:font-normal group-data-[collapsible=icon]:!p-1.5 hover:!bg-zinc-200 [&>svg]:size-5"
          >
            <Link href={link.href}>
              <link.icon className="size-5" />
              <span className="truncate">{link.name}</span>
              <NavLinkLoader />
            </Link>
          </SidebarMenuButton>
        </SidebarMenuItem>
      ))}
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
              className="ml-auto size-4 transition-transform duration-200 ease-[cubic-bezier(0.33,1,0.68,1)] group-data-[state=open]/collapsible:rotate-180 motion-reduce:transition-none"
            />
          </CollapsibleTrigger>
        </SidebarGroupLabel>
        <CollapsibleContent
          className={cn(
            "overflow-hidden data-[state=closed]:animate-collapsible-up data-[state=open]:animate-collapsible-down motion-reduce:animate-none",
            scrollable && "flex min-h-0 flex-1 flex-col",
          )}
        >
          <SidebarGroupContent
            className={
              scrollable
                ? "min-h-0 flex-1 overflow-y-auto [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
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
    <Sidebar
      collapsible="icon"
      {...props}
      className="[&_[data-sidebar=sidebar]]:bg-[#F3F3F4]"
    >
      <AppSidebarHeader />

      <SidebarContent className="gap-2 overflow-hidden">
        <motion.div
          variants={sidebarContainerVariants}
          initial="hidden"
          animate="show"
          className="flex min-h-0 flex-1 flex-col gap-2"
        >
          <motion.div variants={itemVariants}>
            <SidebarGroup className="mt-2 py-1 group-data-[collapsible=icon]:mt-0">
              <SidebarGroupContent>
                <SidebarMenu>
                  <SidebarMenuItem>
                    <SidebarMenuButton
                      asChild
                      tooltip="New Task"
                      className="justify-center rounded-lg bg-zinc-800 font-medium text-white group-data-[collapsible=icon]:justify-start hover:!bg-zinc-900 hover:!text-white"
                    >
                      <Link href="/copilot">
                        <NewTaskIcon />
                        <span className="truncate">New Task</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          </motion.div>

          <motion.div variants={itemVariants}>
            <SidebarGroup className="mt-2 py-1 group-data-[collapsible=icon]:mt-0">
              <SidebarGroupContent>
                <NavMenu links={MAIN_LINKS} leading={<SidebarSearch />} />
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
            className="flex min-h-0 flex-1 flex-col group-data-[collapsible=icon]:hidden"
          >
            <CollapsibleNavGroup label="Recent chats" scrollable>
              {/* Suspense boundary: RecentChats reads useSearchParams(), which
                  Next.js requires to be wrapped to avoid forcing the route to
                  client-side rendering. */}
              <Suspense fallback={null}>
                <RecentChats />
              </Suspense>
            </CollapsibleNavGroup>
          </motion.div>
        </motion.div>
      </SidebarContent>

      <SidebarRail />
    </Sidebar>
  );
}
