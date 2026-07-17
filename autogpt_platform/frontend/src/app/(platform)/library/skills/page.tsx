"use client";

import { useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowLeftIcon } from "@phosphor-icons/react";
import { SkillsPanel } from "@/components/contextual/SkillsPanel/SkillsPanel";

export default function SkillsPage() {
  const router = useRouter();

  useEffect(() => {
    document.title = "AutoPilot skills – AutoGPT Platform";
  }, []);

  function handleGuidedPrompt(prompt: string) {
    router.push(`/copilot#prompt=${encodeURIComponent(prompt)}`);
  }

  return (
    <main className="container min-h-screen space-y-6 pb-20 pt-16 sm:px-8 md:px-12">
      <Link
        href="/library"
        className="inline-flex items-center gap-1 text-sm text-zinc-500 hover:text-zinc-800"
        data-testid="skills-back-to-library"
      >
        <ArrowLeftIcon size={14} weight="bold" />
        Back to Library
      </Link>
      <SkillsPanel onGuidedPrompt={handleGuidedPrompt} />
    </main>
  );
}
