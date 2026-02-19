import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Copilot Styleguide",
  description: "Copilot UI component styleguide",
};

export default function StyleguideLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
