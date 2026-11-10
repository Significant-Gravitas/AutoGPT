"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
  useSidebar,
} from "@/components/molecules/Sidebar/Sidebar";
import { isEditableElement } from "@/lib/platform";
import {
  AddTeamIcon,
  CheckListIcon,
  FlowIcon,
  Folder01Icon,
  GridViewIcon,
  Home01Icon,
  Store01Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { tourTarget, type TourTarget } from "@/lib/tour-targets";
import { motion, useReducedMotion } from "framer-motion";
import Link, { useLinkStatus } from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { ComponentProps, ReactNode, Suspense, useEffect } from "react";
import { getSidebarItemVariants, sidebarContainerVariants } from "./animations";
import { AppSidebarHeader } from "./components/AppSidebarHeader/AppSidebarHeader";
import { CollapsibleNavGroup } from "./components/CollapsibleNavGroup/CollapsibleNavGroup";
import { ExpertsNavSection } from "./components/ExpertsNavSection/ExpertsNavSection";
import { ShortcutHint } from "./components/ShortcutHint/ShortcutHint";
import { SidebarUserActions } from "./components/SidebarUserActions/SidebarUserActions";

type NavLink = {
  name: string;
  href: string;
  icon: IconSvgElement;
  /** Pointed at by the first-run walkthrough. */
  tourTarget?: TourTarget;
};

const MAIN_LINKS: NavLink[] = [
  { name: "Workflows", href: "/library", icon: GridViewIcon },
  { name: "Marketplace", href: "/marketplace", icon: Store01Icon },
  { name: "Build", href: "/build", icon: FlowIcon },
];

const WORKSPACE_LINKS: NavLink[] = [
  {
    name: "Files",
    href: "/artifacts",
    icon: Folder01Icon,
    tourTarget: "files",
  },
];

const TASKS_LINK: NavLink = {
  name: "Tasks",
  href: "/tasks",
  icon: CheckListIcon,
};

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
      tone="sidebar-soft"
      className="ml-auto !size-4 shrink-0"
    />
  );
}

function HomeIcon() {
  const { pending } = useLinkStatus();

  if (pending) {
    return (
      <LoadingSpinner
        size="small"
        tone="sidebar-soft"
        className="!size-4 shrink-0"
      />
    );
  }

  return <Icon icon={Home01Icon} tone="inherit" className="size-4" />;
}

function HomeItem() {
  const pathname = usePathname();

  return (
    <SidebarMenuItem>
      <SidebarMenuButton
        asChild
        size="shell"
        tooltip="Home"
        isActive={isLinkActive(pathname, "/home")}
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

  return (
    <SidebarMenuItem {...(link.tourTarget && tourTarget(link.tourTarget))}>
      <SidebarMenuButton
        asChild
        size="shell"
        tooltip={link.name}
        isActive={isLinkActive(pathname, link.href)}
      >
        <Link href={link.href}>
          <Icon icon={link.icon} tone="inherit" className="size-4" />
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
    <SidebarMenu>
      {leading}
      {links.map((link) => (
        <NavItem key={link.href} link={link} />
      ))}
    </SidebarMenu>
  );
}

type Props = ComponentProps<typeof Sidebar>;

export function AppSidebar(props: Props) {
  const { isLoggedIn } = useAuth();
  const reduceMotion = useReducedMotion();
  const { isMobile } = useSidebar();
  const itemVariants = getSidebarItemVariants(!!reduceMotion);
  const router = useRouter();

  const mainLinks = MAIN_LINKS.filter((link) => link.href !== "/library");

  const workspaceLinks = [
    {
      name: "Team",
      href: "/team",
      icon: AddTeamIcon,
      tourTarget: "team" as const,
    },
    ...(isLoggedIn ? [TASKS_LINK] : []),
    ...WORKSPACE_LINKS,
  ];

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
    <Sidebar collapsible="icon" {...props}>
      <SidebarContent className="scroll-py-16 [scrollbar-width:none] gap-0 overflow-x-hidden overflow-y-auto overscroll-contain [-ms-overflow-style:none] group-data-[collapsible=icon]:scroll-pt-24 group-data-[collapsible=icon]:overflow-y-auto [&::-webkit-scrollbar]:hidden">
        <AppSidebarHeader />

        <motion.div
          variants={sidebarContainerVariants}
          // On mobile the sheet's slide is the entrance; a blurred stagger on
          // top of it delays the items and costs frames on phones.
          initial={isMobile ? false : "hidden"}
          animate="show"
          className="flex shrink-0 flex-col"
        >
          <motion.div variants={itemVariants}>
            <SidebarGroup size="shell" className="mt-0">
              <SidebarGroupContent>
                <NavMenu links={mainLinks} leading={<HomeItem />} />
              </SidebarGroupContent>
            </SidebarGroup>
          </motion.div>

          {workspaceLinks.length > 0 ? (
            <motion.div variants={itemVariants} {...tourTarget("workspace")}>
              <CollapsibleNavGroup label="Workspace">
                <NavMenu links={workspaceLinks} />
              </CollapsibleNavGroup>
            </motion.div>
          ) : null}

          {isLoggedIn && (
            <motion.div
              variants={itemVariants}
              className="group-data-[collapsible=icon]:hidden"
              {...tourTarget("chats")}
            >
              {/* ExpertsNavSection reads useSearchParams(), which Next.js
                  requires to sit under a Suspense boundary. */}
              <Suspense fallback={null}>
                <ExpertsNavSection />
              </Suspense>
            </motion.div>
          )}
        </motion.div>

        <SidebarUserActions />
      </SidebarContent>

      <SidebarRail className="w-2 translate-x-0 group-data-[side=left]:-right-2 group-data-[side=right]:-left-2 after:left-0 group-data-[side=right]:after:right-0 group-data-[side=right]:after:left-auto sm:hidden md:flex" />
    </Sidebar>
  );
}
