"use client";

import * as React from "react";
import { AgentTable } from "@/components/agptui/AgentTable";
import { Button } from "@/components/agptui/Button";
import { Separator } from "@/components/ui/separator";
import { StatusType } from "@/components/agptui/Status";
import { PublishAgentPopout } from "@/components/agptui/composite/PublishAgentPopout";
import { useCallback, useEffect, useState } from "react";
import {
  StoreSubmissionsResponse,
  StoreSubmissionRequest,
} from "@/lib/autogpt-server-api/types";
import useSupabase from "@/hooks/useSupabase";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import AutogptButton from "@/components/agptui/AutogptButton";

export default function Page({}: {}) {
  const { supabase } = useSupabase();
  const api = useBackendAPI();
  const [submissions, setSubmissions] = useState<StoreSubmissionsResponse>();
  const [openPopout, setOpenPopout] = useState<boolean>(false);
  const [submissionData, setSubmissionData] =
    useState<StoreSubmissionRequest>();
  const [popoutStep, setPopoutStep] = useState<"select" | "info" | "review">(
    "info",
  );

  const fetchData = useCallback(async () => {
    try {
      const submissions = await api.getStoreSubmissions();
      setSubmissions(submissions);
    } catch (error) {
      console.error("Error fetching submissions:", error);
    }
  }, [api]);

  useEffect(() => {
    if (!supabase) {
      return;
    }
    fetchData();
  }, [supabase, fetchData]);

  const onEditSubmission = useCallback((submission: StoreSubmissionRequest) => {
    setSubmissionData(submission);
    setPopoutStep("review");
    setOpenPopout(true);
  }, []);

  const onDeleteSubmission = useCallback(
    (submission_id: string) => {
      if (!supabase) {
        return;
      }
      api.deleteStoreSubmission(submission_id);
      fetchData();
    },
    [api, supabase, fetchData],
  );

  const onOpenPopout = useCallback(() => {
    setPopoutStep("select");
    setOpenPopout(true);
  }, []);

  return (
    <main className="flex-1 space-y-7.5 pb-8">
      {/* Title */}
      <h1 className="font-poppins text-[1.75rem] font-medium leading-[2.5rem] text-zinc-500">
        Agent dashboard
      </h1>

      {/* Content  */}
      <section className="space-y-8">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="font-poppins text-base font-medium text-zinc-800">
              Submit a New Agent
            </h2>

            <p className="font-sans text-base font-normal text-zinc-600">
              Select from the list of agents you currently have, or upload from
              your local machine.
            </p>
          </div>

          <PublishAgentPopout
            trigger={
              <AutogptButton onClick={onOpenPopout} variant="outline">
                Add to Library
              </AutogptButton>
            }
            openPopout={openPopout}
            inputStep={popoutStep}
            submissionData={submissionData}
          />
        </div>

        <Separator className="bg-neutral-300" />

        <div className="space-y-3">
          <h2 className="font-poppins text-base font-medium text-zinc-800">
            Your uploaded agents
          </h2>

          {submissions && (
            <AgentTable
              agents={
                submissions?.submissions.map((submission, index) => ({
                  id: index,
                  agent_id: submission.agent_id,
                  agent_version: submission.agent_version,
                  sub_heading: submission.sub_heading,
                  date_submitted: submission.date_submitted,
                  agentName: submission.name,
                  description: submission.description,
                  imageSrc: submission.image_urls || [""],
                  dateSubmitted: new Date(
                    submission.date_submitted,
                  ).toLocaleDateString(),
                  status: submission.status.toLowerCase() as StatusType,
                  runs: submission.runs,
                  rating: submission.rating,
                })) || []
              }
              onEditSubmission={onEditSubmission}
              onDeleteSubmission={onDeleteSubmission}
            />
          )}
        </div>
      </section>
    </main>
  );
}
