"use client";

import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { SidebarMenuButton, SidebarMenuItem } from "@/components/ui/sidebar";
import { isEditableElement } from "@/lib/platform";
import PiPencilEditBoxSolidStroke from "@/components/icons/pika/vendor/PiPencilEditBoxSolidStroke";
import Link, { useLinkStatus } from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect } from "react";
import { SidebarShortcutHint } from "../SidebarShortcutHint/SidebarShortcutHint";

const NEW_TASK_HREF = "/copilot";

// Rendered inside the New Task <Link> — swap the icon for a spinner while
// navigation to /copilot is pending, then back to the icon once it lands.
function NewTaskIcon() {
  const { pending } = useLinkStatus();

  if (pending) {
    return <LoadingSpinner size="small" className="shrink-0" />;
  }

  return <PiPencilEditBoxSolidStroke className="size-4" />;
}

export function NewTaskItem() {
  const pathname = usePathname();
  const router = useRouter();
  const isActive = pathname === NEW_TASK_HREF;

  useEffect(() => {
    function handleShortcut(event: KeyboardEvent) {
      if (event.repeat) return;
      if (event.key.toLocaleLowerCase() !== "o") return;
      if (!event.shiftKey) return;
      if (!event.metaKey && !event.ctrlKey) return;
      if (isEditableElement(document.activeElement)) return;
      event.preventDefault();
      router.push(NEW_TASK_HREF);
    }

    document.addEventListener("keydown", handleShortcut);
    return () => document.removeEventListener("keydown", handleShortcut);
  }, [router]);

  return (
    <SidebarMenuItem>
      <SidebarMenuButton
        asChild
        tooltip="New Task"
        isActive={isActive}
        className="h-auto rounded-lg p-2 font-normal data-[active=true]:!bg-zinc-100 data-[active=true]:font-normal group-data-[collapsible=icon]:justify-center group-data-[collapsible=icon]:!gap-0 group-data-[collapsible=icon]:!p-1.5 hover:!bg-zinc-100 [&>svg]:size-4"
      >
        <Link href={NEW_TASK_HREF}>
          <NewTaskIcon />
          <span className="truncate group-data-[collapsible=icon]:hidden">
            New Task
          </span>
          <SidebarShortcutHint
            mac={["⇧", "⌘", "O"]}
            other={["⇧", "Ctrl", "O"]}
          />
        </Link>
      </SidebarMenuButton>
    </SidebarMenuItem>
  );
}
