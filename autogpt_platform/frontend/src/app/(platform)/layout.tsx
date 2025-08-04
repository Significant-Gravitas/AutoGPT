import { Navbar } from "@/components/layout/Navbar/Navbar";
import { ReactNode } from "react";

export default function PlatformLayout({ children }: { children: ReactNode }) {
  return (
    <>
      {/* FRONTEND-TODO: We need to add different color for different pages */}
      <Navbar />
      <main className="flex-1 bg-lightGrey">{children}</main>
    </>
  );
}
