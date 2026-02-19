import { redirect } from "next/navigation";

import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "AutoGPT Platform",
  description: "AutoGPT Platform",
};

export default function Page() {
  redirect("/copilot");
}
