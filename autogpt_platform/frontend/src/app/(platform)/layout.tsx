import { Navbar } from "@/components/layout/Navbar/Navbar";
import { ReactNode } from "react";

export default function PlatformLayout({ children }: { children: ReactNode }) {
  return (
    <main className="flex h-screen w-full flex-col">
      <Navbar />
      <section className="flex-1">{children}</section>
    </main>
  );
}
