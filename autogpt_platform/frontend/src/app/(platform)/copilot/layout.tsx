import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Copilot",
  description: "Chat with your AI copilot",
};

export default function CopilotLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
