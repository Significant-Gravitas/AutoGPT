"use client";

import { boardHref } from "@/app/(platform)/home/board-helpers";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import { KanbanIcon } from "@hugeicons/core-free-icons";
import { motion, type Variants } from "framer-motion";
import Link from "next/link";
import { useNavItemClassName } from "../../useNavItemClassName";
import { CollapsibleNavGroup } from "../CollapsibleNavGroup/CollapsibleNavGroup";
import { NavRailItem } from "../NavRailItem/NavRailItem";
import { NewProjectItem } from "./components/NewProjectItem/NewProjectItem";
import { useProjectsNavSection } from "./useProjectsNavSection";

interface Props {
  itemVariants: Variants;
}

export function ProjectsNavSection({ itemVariants }: Props) {
  const navItemClassName = useNavItemClassName();
  const {
    projects,
    isAtLimit,
    isCreating,
    isOnProject,
    isProjectActive,
    createProject,
    isOpen,
    setIsOpen,
    groupRef,
    jumpToGroup,
  } = useProjectsNavSection();

  return (
    <motion.div variants={itemVariants}>
      <div className="hidden group-data-[collapsible=icon]:block">
        <NavRailItem
          icon={KanbanIcon}
          label="Projects"
          isActive={isOnProject}
          onClick={jumpToGroup}
        />
      </div>

      <div ref={groupRef} className="group-data-[collapsible=icon]:hidden">
        <CollapsibleNavGroup
          label="Projects"
          open={isOpen}
          onOpenChange={setIsOpen}
        >
          <SidebarMenu>
            {projects.map((project) => (
              <SidebarMenuItem key={project.id}>
                <SidebarMenuButton
                  asChild
                  isActive={isProjectActive(project.id)}
                  className={navItemClassName}
                >
                  <Link href={boardHref(project.id)}>
                    <Icon
                      icon={KanbanIcon}
                      className="size-4 text-sidebar-foreground/90"
                    />
                    <span className="truncate">{project.name}</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
            ))}
            <NewProjectItem
              isAtLimit={isAtLimit}
              isCreating={isCreating}
              onCreate={createProject}
            />
          </SidebarMenu>
        </CollapsibleNavGroup>
      </div>
    </motion.div>
  );
}
