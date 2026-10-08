import { useSidebar } from "@/components/ui/sidebar";

// Kobra's useSidebar throws outside a SidebarProvider; the context read has
// already happened by then, so catching keeps the hook order stable.
export function useOptionalSidebar() {
  try {
    return useSidebar();
  } catch {
    return null;
  }
}
