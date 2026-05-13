import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Shared Agent Run - AutoGPT",
  description: "View shared agent run results",
  robots: "noindex, nofollow",
};

// Passthrough.  Each share page owns its own viewport — the chat
// viewer needs full-bleed h-screen, while the execution share page
// renders its own logo header + ``container`` wrapper inline.
// Wrapping everything here forced ``h-screen`` content to overflow
// the document body, producing a page-level scrollbar on top of
// the chat's internal one.
export default function ShareLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}
