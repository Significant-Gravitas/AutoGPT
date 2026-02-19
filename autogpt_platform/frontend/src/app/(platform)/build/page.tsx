import type { Metadata } from "next";
import { BuilderContent } from "./BuilderContent";

export const metadata: Metadata = {
  title: "Build",
  description: "Build your agent",
};

export default function BuilderPage() {
  return <BuilderContent />;
}
