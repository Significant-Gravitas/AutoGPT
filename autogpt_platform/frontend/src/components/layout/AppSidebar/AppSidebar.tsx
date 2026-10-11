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
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { isEditableElement } from "@/lib/platform";
import { cn } from "@/lib/utils";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { motion, useReducedMotion } from "framer-motion";
import Link, { useLinkStatus } from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { ComponentProps, ReactNode, Suspense, useEffect } from "react";
import { getSidebarItemVariants, sidebarContainerVariants } from "./animations";
import { AppSidebarHeader } from "./components/AppSidebarHeader/AppSidebarHeader";
import { RecentChats } from "./components/RecentChats/RecentChats";
import { ShortcutHint } from "./components/ShortcutHint/ShortcutHint";
import { SidebarUserActions } from "./components/SidebarUserActions/SidebarUserActions";
import {
  ArrowDown01Icon,
  FlowIcon,
  Folder01Icon,
  GridViewIcon,
  Home10Icon,
  Store01Icon,
  AddTeamIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { Icon } from "@/components/atoms/Icon/Icon";

type NavLink = {
  name: string;
  href: string;
  icon: IconSvgElement;
};

const MAIN_LINKS: NavLink[] = [
  { name: "Agents", href: "/library", icon: GridViewIcon },
  { name: "Marketplace", href: "/marketplace", icon: Store01Icon },
  { name: "Build", href: "/build", icon: FlowIcon },
];

const WORKSPACE_LINKS: NavLink[] = [
  { name: "Files", href: "/artifacts", icon: Folder01Icon },
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
      className="ml-auto !size-4 shrink-0 text-sidebar-foreground/90 group-data-[collapsible=icon]:!size-4.5"
    />
  );
}

function HomeIcon() {
  const { pending } = useLinkStatus();

  if (pending) {
    return (
      <LoadingSpinner
        size="small"
        className="!size-4 shrink-0 text-sidebar-foreground/90 group-data-[collapsible=icon]:!size-4.5"
      />
    );
  }

  return (
    <Icon
      icon={Home10Icon}
      className="size-4 text-sidebar-foreground/90 group-data-[collapsible=icon]:size-4.5"
    />
  );
}

// The stronger active state ships with the brain-dump experience.
function useNavItemClassName() {
  const isBrainDumpEnabled = useGetFlag(Flag.ONBOARDING_BRAIN_DUMP);
  return cn(
    "h-auto rounded-xl p-2 pl-3 font-normal data-[active=true]:font-normal group-data-[collapsible=icon]:!p-1.5 hover:!bg-zinc-100 [&>svg]:size-4 group-data-[collapsible=icon]:[&>svg]:size-4.5",
    isBrainDumpEnabled
      ? "data-[active=true]:!bg-zinc-200 data-[active=true]:hover:!bg-zinc-200"
      : "data-[active=true]:!bg-zinc-100",
  );
}

function HomeItem() {
  const pathname = usePathname();
  const navItemClassName = useNavItemClassName();

  return (
    <SidebarMenuItem>
      <SidebarMenuButton
        asChild
        tooltip="Home"
        isActive={isLinkActive(pathname, "/home")}
        className={navItemClassName}
      >
        <Link href="/home">
          <HomeIcon />
          <span className="truncate">Home</span>
          <ShortcutHint letter="O" />
        </Link>
      </SidebarMenuButton>
    </SidebarMenuItem>
  );
}

interface NavItemProps {
  link: NavLink;
}

function NavItem({ link }: NavItemProps) {
  const pathname = usePathname();
  const navItemClassName = useNavItemClassName();

  return (
    <SidebarMenuItem>
      <SidebarMenuButton
        asChild
        tooltip={link.name}
        isActive={isLinkActive(pathname, link.href)}
        className={navItemClassName}
      >
        <Link href={link.href}>
          <Icon
            icon={link.icon}
            className="size-4 text-sidebar-foreground/90 group-data-[collapsible=icon]:size-4.5"
          />
          <span className="truncate">{link.name}</span>
          <NavLinkLoader />
        </Link>
      </SidebarMenuButton>
    </SidebarMenuItem>
  );
}

function NavMenu({
  links,
  leading,
}: {
  links: NavLink[];
  leading?: ReactNode;
}) {
  return (
    <SidebarMenu className="group-data-[collapsible=icon]:gap-1">
      {leading}
      {links.map((link) => (
        <NavItem key={link.href} link={link} />
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
        <SidebarGroupLabel
          asChild
          className="text-[13px] font-medium text-zinc-500 group-data-[collapsible=icon]:hidden"
        >
          <CollapsibleTrigger>
            {label}
            <Icon
              icon={ArrowDown01Icon}
              className="ease-[cubic-bezier(0.33,1,0.68,1)] ml-auto size-4 text-sidebar-foreground/90 transition-transform duration-200 group-data-[collapsible=icon]:size-4.5 group-data-[state=open]/collapsible:rotate-180 motion-reduce:transition-none"
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
  const { isLoggedIn } = useAuth();
  const reduceMotion = useReducedMotion();
  const itemVariants = getSidebarItemVariants(!!reduceMotion);
  const router = useRouter();
  const isHireExpertsEnabled = useGetFlag(Flag.HIRE_EXPERTS);
  const mainLinks = isHireExpertsEnabled
    ? MAIN_LINKS.filter((link) => link.href !== "/library")
    : MAIN_LINKS;
  const filesEnabled = useGetFlag(Flag.ARTIFACTS_PAGE);
  const workspaceLinks = (
    isHireExpertsEnabled
      ? [{ name: "Team", href: "/team", icon: AddTeamIcon }, ...WORKSPACE_LINKS]
      : WORKSPACE_LINKS
  ).filter((link) => link.href !== "/artifacts" || filesEnabled);

  useEffect(() => {
    function handleNewTaskShortcut(event: KeyboardEvent) {
      if (event.repeat) return;
      if (!event.key || event.key.toLocaleLowerCase() !== "o") return;
      if (!event.metaKey && !event.ctrlKey) return;
      if (!event.shiftKey) return;
      if (isEditableElement(document.activeElement)) return;
      event.preventDefault();
      router.push("/home");
    }

    document.addEventListener("keydown", handleNewTaskShortcut);
    return () => document.removeEventListener("keydown", handleNewTaskShortcut);
  }, [router]);

  return (
    <Sidebar
      collapsible="icon"
      {...props}
      className="[&_[data-sidebar=sidebar]]:bg-[#fafafa]"
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
            <SidebarGroup className="mt-0 py-1">
              <SidebarGroupContent>
                <NavMenu links={mainLinks} leading={<HomeItem />} />
              </SidebarGroupContent>
            </SidebarGroup>
          </motion.div>

          {workspaceLinks.length > 0 ? (
            <motion.div variants={itemVariants}>
              <CollapsibleNavGroup label="Workspace">
                <NavMenu links={workspaceLinks} />
              </CollapsibleNavGroup>
            </motion.div>
          ) : null}

          {isLoggedIn && (
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
          )}
        </motion.div>
      </SidebarContent>

      <SidebarUserActions />

      <SidebarRail />
    </Sidebar>
  );
}
