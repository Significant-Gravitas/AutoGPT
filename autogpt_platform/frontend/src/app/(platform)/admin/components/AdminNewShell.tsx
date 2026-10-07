"use client";

import { ScrollArea } from "@/components/atoms/ScrollArea/ScrollArea";
import { motion, useReducedMotion } from "framer-motion";
import { usePathname } from "next/navigation";
import { ReactNode } from "react";

import { AdminMobileNav } from "./AdminMobileNav/AdminMobileNav";
import { AdminSidebar } from "./AdminSidebar/AdminSidebar";

// Mirrors the new /settings shell: own sidebar with a Back link, no top
// Navbar (PlatformChrome renders the bare shell for /admin under the new
// layout).
export function AdminNewShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const reduceMotion = useReducedMotion();

  return (
    <div className="flex h-full w-full overflow-hidden bg-zinc-50">
      <AdminSidebar />
      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <AdminMobileNav />
        <main className="flex-1 overflow-hidden bg-zinc-50">
          <ScrollArea className="h-full">
            <motion.div
              key={pathname}
              initial={reduceMotion ? { opacity: 0 } : { opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.28, ease: [0, 0, 0.2, 1] as const }}
              className="mx-auto w-full max-w-[1360px] px-4 pt-2 pb-8 md:px-6 md:pt-6"
            >
              {children}
            </motion.div>
          </ScrollArea>
        </main>
      </div>
    </div>
  );
}
