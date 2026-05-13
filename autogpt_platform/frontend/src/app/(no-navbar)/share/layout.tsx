import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Shared Agent Run - AutoGPT",
  description: "View shared agent run results",
  robots: "noindex, nofollow",
};

// Logo header at the top, full-bleed children below.  Children get
// ``flex-1 min-h-0`` so a child that wants the full remaining height
// (like the chat viewer) can use ``h-full`` without overflowing the
// document.  Pages that want a container/max-width wrapper add it
// themselves (e.g. the execution-share page wraps its content in
// ``container mx-auto px-4 py-8``).
export default function ShareLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex h-screen w-full flex-col bg-background">
      <header className="shrink-0 border-b border-border bg-background">
        <div className="container mx-auto flex justify-center px-4 py-4">
          <Link href="/" className="inline-block">
            <Image
              src="/autogpt-logo-dark-bg.png"
              alt="AutoGPT"
              width={120}
              height={54}
              className="hidden h-8 w-auto dark:block"
            />
            <Image
              src="/autogpt-logo-light-bg.png"
              alt="AutoGPT"
              width={120}
              height={54}
              className="block h-8 w-auto dark:hidden"
              priority
            />
          </Link>
        </div>
      </header>
      <main className="min-h-0 flex-1 overflow-hidden">{children}</main>
    </div>
  );
}
