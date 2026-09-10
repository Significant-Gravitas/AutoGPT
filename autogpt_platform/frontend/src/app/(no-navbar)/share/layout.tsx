import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Shared Agent Run - AutoGPT",
  description: "View shared agent run results",
  robots: "noindex, nofollow",
};

// Passthrough.  Brand chrome lives in ``./components/ShareHeader``
// which both share pages import — consistency comes from the shared
// component, not from this file.  Pages own their own viewport/scroll
// behaviour so the chat viewer can be full-bleed while the execution
// viewer scrolls inside its own container.
export default function ShareLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}
