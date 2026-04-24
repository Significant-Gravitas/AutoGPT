"use client";

import { ReactNode } from "react";
import { usePathname } from "next/navigation";
import { motion, useReducedMotion } from "framer-motion";
import { SettingsSidebar } from "./components/SettingsSidebar/SettingsSidebar";
import { SettingsMobileNav } from "./components/SettingsMobileNav/SettingsMobileNav";

export default function SettingsLayout({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const reduceMotion = useReducedMotion();

  return (
    <div className="flex h-full w-full overflow-hidden bg-[#F9F9FA]">
      <SettingsSidebar />
      <div className="flex flex-1 flex-col overflow-hidden">
        <SettingsMobileNav />
        <main className="flex-1 overflow-y-auto bg-[#F9F9FA] px-4 pt-2 md:px-[111px] md:pt-[39px]">
          <motion.div
            key={pathname}
            initial={reduceMotion ? { opacity: 0 } : { opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.28, ease: [0, 0, 0.2, 1] as const }}
          >
            {children}
          </motion.div>
        </main>
      </div>
    </div>
  );
}
