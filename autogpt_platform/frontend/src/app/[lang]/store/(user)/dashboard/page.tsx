import * as React from "react";
import { AgentTable } from "@/components/agptui/AgentTable";
import { Button } from "@/components/agptui/Button";
import { Separator } from "@/components/ui/separator";
import AutoGPTServerAPIServerSide from "@/lib/autogpt-server-api";
import { createServerClient } from "@/lib/supabase/server";
import { StatusType } from "@/components/agptui/Status";
import { PublishAgentPopout } from "@/components/agptui/composite/PublishAgentPopout";

async function getDashboardData() {
  // Get the supabase client first
  const supabase = createServerClient();
  const {
    data: { session },
  } = await supabase.auth.getSession();

  if (!session) {
    console.warn("--- No session found in profile page");
    return { profile: null };
  }

  // Create API client with the same supabase instance
  const api = new AutoGPTServerAPIServerSide(
    process.env.NEXT_PUBLIC_AGPT_SERVER_URL,
    process.env.NEXT_PUBLIC_AGPT_WS_SERVER_URL,
    supabase, // Pass the supabase client instance
  );

  try {
    const submissions = await api.getStoreSubmissions();
    return {
      submissions,
    };
  } catch (error) {
    console.error("Error fetching profile:", error);
    return {
      profile: null,
    };
  }
}

export default async function Page({
  params: { lang },
}: {
  params: { lang: string };
}) {
  const { submissions } = await getDashboardData();

  return (
    <main className="flex-1 px-6 py-8 md:px-10">
      {/* Header Section */}
      <div className="mb-8 flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="font-neue text-3xl font-medium leading-9 tracking-tight text-neutral-900 dark:text-neutral-100">
            Submit a New Agent
          </h1>
          <p className="mt-2 font-neue text-sm text-[#707070] dark:text-neutral-400">
            Select from the list of agents you currently have, or upload from
            your local machine.
          </p>
        </div>
        <PublishAgentPopout
          trigger={
            <Button variant="default" size="lg">
              Create New Agent
            </Button>
          }
        />
      </div>

      <Separator className="mb-8" />

      {/* Agents Section */}
      <div>
        <h2 className="mb-4 text-xl font-bold text-neutral-900 dark:text-neutral-100">Your Agents</h2>
        <AgentTable
          agents={
            submissions?.submissions.map((submission, index) => ({
              id: index,
              agentName: submission.name,
              description: submission.description,
              imageSrc: submission.image_urls[0] || "",
              dateSubmitted: new Date(
                submission.date_submitted,
              ).toLocaleDateString(),
              status: submission.status.toLowerCase() as StatusType,
            })) || []
          }
        />
      </div>
    </main>
  );
}
