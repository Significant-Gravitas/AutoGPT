"use client";

import { Text } from "@/components/atoms/Text/Text";
import { usePathname } from "next/navigation";
import { ChatSessionList } from "./ChatSessionList";

export function SidebarDynamicContent() {
  const pathname = usePathname();

  if (pathname.startsWith("/copilot")) {
    return <ChatSessionList />;
  }

  if (pathname.startsWith("/library")) {
    return <GenericSidebarContent label="Library" />;
  }

  if (pathname.startsWith("/marketplace")) {
    return <GenericSidebarContent label="Marketplace" />;
  }

  if (pathname.startsWith("/build")) {
    return <GenericSidebarContent label="Builder" />;
  }

  return null;
}

function GenericSidebarContent({ label }: { label: string }) {
  return (
    <div className="px-3 py-2">
      <Text variant="h3" size="body-medium">
        {label}
      </Text>
    </div>
  );
}
