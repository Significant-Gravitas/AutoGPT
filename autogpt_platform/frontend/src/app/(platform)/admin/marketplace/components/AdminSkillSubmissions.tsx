"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { useAdminSkillSubmissions } from "./useAdminSkillSubmissions";

export function AdminSkillSubmissions() {
  const {
    submissions,
    isLoading,
    isUnavailable,
    isReviewing,
    approve,
    reject,
  } = useAdminSkillSubmissions();

  if (isLoading) {
    return <div className="py-6 text-center">Loading skill submissions…</div>;
  }

  if (isUnavailable) {
    return (
      <Text variant="body" className="!text-zinc-500">
        Skill submissions are unavailable right now.
      </Text>
    );
  }

  if (submissions.length === 0) {
    return (
      <Text variant="body" className="!text-zinc-500">
        No skill submissions are waiting for review.
      </Text>
    );
  }

  return (
    <ul className="flex flex-col gap-3" data-testid="admin-skill-submissions">
      {submissions.map((submission) => (
        <li
          key={submission.skill_listing_version_id}
          className="flex flex-wrap items-start justify-between gap-3 rounded-xl border border-zinc-200 bg-white p-4"
        >
          <div className="min-w-0">
            <Text variant="body-medium">
              {submission.name}{" "}
              <span className="text-zinc-400">v{submission.version}</span>
            </Text>
            <Text variant="small" className="!text-zinc-500">
              {submission.description}
            </Text>
            <Text variant="small" className="!mt-1 !text-zinc-400">
              /{submission.slug} · {submission.categories.join(", ")}
              {submission.required_providers.length > 0
                ? ` · works with ${submission.required_providers.join(", ")}`
                : ""}
            </Text>
          </div>
          <div className="flex flex-shrink-0 gap-2">
            <Button
              variant="secondary"
              size="small"
              disabled={isReviewing}
              onClick={() => reject(submission.skill_listing_version_id)}
              data-testid={`reject-${submission.skill_listing_version_id}`}
            >
              Reject
            </Button>
            <Button
              variant="primary"
              size="small"
              disabled={isReviewing}
              onClick={() => approve(submission.skill_listing_version_id)}
              data-testid={`approve-${submission.skill_listing_version_id}`}
            >
              Approve
            </Button>
          </div>
        </li>
      ))}
    </ul>
  );
}
