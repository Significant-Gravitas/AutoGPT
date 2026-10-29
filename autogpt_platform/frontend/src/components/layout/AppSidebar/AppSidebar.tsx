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
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { isEditableElement } from "@/lib/platform";
import {
  AddTeamIcon,
  FlowIcon,
  Folder01Icon,
  GridViewIcon,
  Home10Icon,
  Store01Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { tourTarget, type TourTarget } from "@/lib/tour-targets";
import { motion, useReducedMotion } from "framer-motion";
import Link, { useLinkStatus } from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { ComponentProps, ReactNode, useEffect } from "react";
import { getSidebarItemVariants, sidebarContainerVariants } from "./animations";
import { AppSidebarHeader } from "./components/AppSidebarHeader/AppSidebarHeader";
import { ChatsNavSection } from "./components/ChatsNavSection/ChatsNavSection";
import { CollapsibleNavGroup } from "./components/CollapsibleNavGroup/CollapsibleNavGroup";
import { ShortcutHint } from "./components/ShortcutHint/ShortcutHint";
import { SidebarUserActions } from "./components/SidebarUserActions/SidebarUserActions";
import { useNavItemClassName } from "./useNavItemClassName";

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
    <SidebarMenuItem {...(link.tourTarget && tourTarget(link.tourTarget))}>
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
      <SidebarContent className="scroll-py-16 gap-0 overflow-y-auto overflow-x-hidden overscroll-contain [-ms-overflow-style:none] [scrollbar-width:none] group-data-[collapsible=icon]:scroll-pt-24 group-data-[collapsible=icon]:overflow-y-auto [&::-webkit-scrollbar]:hidden">
        <AppSidebarHeader />

        <motion.div
          variants={sidebarContainerVariants}
          // On mobile the sheet's slide is the entrance; a blurred stagger on
          // top of it delays the items and costs frames on phones.
          initial={isMobile ? false : "hidden"}
          animate="show"
          className="flex shrink-0 flex-col gap-2"
        >
          <motion.div variants={itemVariants}>
            <SidebarGroup className="mt-0 py-1">
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

          {isLoggedIn && <ChatsNavSection itemVariants={itemVariants} />}
        </motion.div>

        <SidebarUserActions />
      </SidebarContent>

      <SidebarRail className="w-2 translate-x-0 after:left-0 group-data-[side=left]:-right-2 group-data-[side=right]:-left-2 group-data-[side=right]:after:left-auto group-data-[side=right]:after:right-0 sm:hidden md:flex" />
    </Sidebar>
  );
}
