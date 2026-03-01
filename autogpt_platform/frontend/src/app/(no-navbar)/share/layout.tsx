import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Shared Agent Run - AutoGPT",
  description: "View shared agent run results",
  robots: "noindex, nofollow",
};

export default function ShareLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-background">
        <div className="container mx-auto flex justify-end px-4 py-4">
          <Link href="/login" className="inline-block">
            <Image
              src="/autogpt-logo-dark-bg.png"
              alt="AutoGPT"
              width={120}
              height={54}
              className="hidden h-8 w-auto dark:block"
              priority
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
      <div className="container mx-auto px-4 py-8">{children}</div>
    </div>
  );
}
