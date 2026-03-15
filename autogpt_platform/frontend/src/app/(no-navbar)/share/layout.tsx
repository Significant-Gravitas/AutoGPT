import type { Metadata } from "next";

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
      <div className="container mx-auto px-4 py-8">{children}</div>
    </div>
  );
}
