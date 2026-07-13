"use client";

import { useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowLeftIcon } from "@phosphor-icons/react";
import { SchedulesPanel } from "@/components/contextual/SchedulesPanel/SchedulesPanel";

export default function FollowupsPage() {
  const router = useRouter();

  useEffect(() => {
    document.title = "Scheduled – AutoGPT Platform";
  }, []);

  function handleGuidedPrompt(prompt: string) {
    router.push(`/copilot#prompt=${encodeURIComponent(prompt)}`);
  }

  return (
    <main className="container min-h-screen space-y-6 pb-20 pt-16 sm:px-8 md:px-12">
      <Link
        href="/library"
        className="inline-flex items-center gap-1 text-sm text-zinc-500 hover:text-zinc-800"
        data-testid="followups-back-to-library"
      >
        <ArrowLeftIcon size={14} weight="bold" />
        Back to Library
      </Link>
      <SchedulesPanel onGuidedPrompt={handleGuidedPrompt} />
    </main>
  );
}
