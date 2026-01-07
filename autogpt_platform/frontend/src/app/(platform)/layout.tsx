import { Navbar } from "@/components/layout/Navbar/Navbar";
import { ReactNode } from "react";
import { AdminImpersonationBanner } from "./admin/components/AdminImpersonationBanner";
import { PlatformLayoutContent } from "./PlatformLayoutContent";

export default function PlatformLayout({ children }: { children: ReactNode }) {
  return (
    <PlatformLayoutContent>
      <Navbar />
      <AdminImpersonationBanner />
      {children}
    </PlatformLayoutContent>
  );
}
