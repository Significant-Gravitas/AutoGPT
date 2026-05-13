import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Shared Agent Run - AutoGPT",
  description: "View shared agent run results",
  robots: "noindex, nofollow",
};

// Shared chrome for every ``/share/...`` route: AutoGPT logo header
// up top, full-height main body underneath.  Each page renders its
// own page-specific meta row inside the body — the layout owns
// brand consistency, the pages own their domain detail.
//
// The body uses ``flex-1 min-h-0 overflow-hidden`` so pages that
// want full-bleed (chat viewer, ``h-full``) and pages that want
// internal scrolling (execution viewer, container + scroll) both
// work without leaking a document-level scrollbar.
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
