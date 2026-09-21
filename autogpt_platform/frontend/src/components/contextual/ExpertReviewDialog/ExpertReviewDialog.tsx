"use client";

import type { ExpertPackageIssue } from "@/app/api/__generated__/models/expertPackageIssue";
import type { ExpertPackagePreview } from "@/app/api/__generated__/models/expertPackagePreview";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { cn } from "@/lib/utils";
import { Alert02Icon } from "@hugeicons/core-free-icons";
import { SkillRow } from "./components/SkillRow";
import { WorkflowRow } from "./components/WorkflowRow";
import type { ExpertImportEdits } from "./helpers";
import { useExpertReviewDialog } from "./useExpertReviewDialog";

interface Props {
  mode: "import" | "publish";
  open: boolean;
  preview: ExpertPackagePreview | null;
  isSubmitting: boolean;
  onClose: () => void;
  onConfirm: (edits: ExpertImportEdits) => void;
}

/** What a `.expert.zip` would become, before it becomes it. Read-only about
 *  everything the file decides; the user only renames it and leaves parts out.
 *  Not a `<form>`, so Enter in the name field can never submit an import.
 *
 *  Publish mode is a confirmation, not an editor: the publish route builds the
 *  package from the stored expert and takes no edits, so an editable control
 *  here could only lie about what reaches the marketplace. */
export function ExpertReviewDialog({
  mode,
  open,
  preview,
  isSubmitting,
  onClose,
  onConfirm,
}: Props) {
  const {
    draft,
    setName,
    toggleSkill,
    toggleWorkflow,
    toggleSchedule,
    edits,
    blockingReason,
  } = useExpertReviewDialog({ open, preview });

  const identity = preview?.manifest.identity;
  const isImport = mode === "import";

  function handleOpenChange(nextOpen: boolean) {
    if (!nextOpen && !isSubmitting) onClose();
  }

  return (
    <Dialog
      variant="compact"
      controlled={{ isOpen: open, set: handleOpenChange }}
      styling={{ maxWidth: "34rem" }}
      title={isImport ? "Review before importing" : "Review before publishing"}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-4">
          <IssueBanner issues={preview?.errors ?? []} tone="error" />
          <IssueBanner issues={preview?.warnings ?? []} tone="warning" />

          {isImport ? (
            <Input
              id="expert-review-name"
              label="Name"
              value={draft.name}
              onChange={(event) => setName(event.target.value)}
            />
          ) : (
            <Text variant="large-medium" className="text-zinc-900">
              {draft.name}
            </Text>
          )}

          {/* Everything else about the expert is what the file says. */}
          <Text variant="small" className="text-zinc-500">
            {[identity?.role, identity?.tagline].filter(Boolean).join(" · ")}
          </Text>

          {(preview?.skills ?? []).length > 0 ? (
            <Section title="Skills">
              {(preview?.skills ?? []).map((skill) => (
                <SkillRow
                  key={skill.slug}
                  skill={skill}
                  isRemoved={draft.removedSkillSlugs.includes(skill.slug)}
                  onToggle={() => toggleSkill(skill.slug)}
                  readOnly={!isImport}
                />
              ))}
            </Section>
          ) : null}

          {(preview?.workflows ?? []).length > 0 ? (
            <Section title="Workflows">
              {(preview?.workflows ?? []).map((workflow) => (
                <WorkflowRow
                  key={workflow.index}
                  workflow={workflow}
                  isRemoved={draft.removedWorkflowIndices.includes(
                    workflow.index,
                  )}
                  isScheduled={draft.scheduledIndices.includes(workflow.index)}
                  onToggle={() => toggleWorkflow(workflow.index)}
                  onToggleSchedule={() => toggleSchedule(workflow.index)}
                  mode={mode}
                />
              ))}
            </Section>
          ) : null}

          {blockingReason ? (
            <Text variant="small" className="text-zinc-500">
              {blockingReason}
            </Text>
          ) : null}
        </div>

        <Dialog.Footer>
          <Button
            variant="secondary"
            size="small"
            disabled={isSubmitting}
            onClick={() => handleOpenChange(false)}
          >
            Cancel
          </Button>
          <Button
            variant="primary"
            size="small"
            loading={isSubmitting}
            disabled={Boolean(blockingReason) || !preview}
            onClick={() => onConfirm(edits)}
          >
            {isImport ? "Import expert" : "Publish"}
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section aria-label={title} className="flex flex-col gap-2">
      <Text variant="small-medium" className="text-zinc-500">
        {title}
      </Text>
      <ul className="flex flex-col gap-2">{children}</ul>
    </section>
  );
}

// Hand-rolled rather than the design system's Alert, which is lucide-based and
// carries a dismiss affordance these two banners must not have.
function IssueBanner({
  issues,
  tone,
}: {
  issues: ExpertPackageIssue[];
  tone: "error" | "warning";
}) {
  if (issues.length === 0) return null;

  return (
    <div
      role={tone === "error" ? "alert" : "status"}
      className={cn(
        "flex gap-2 rounded-lg px-3 py-2 ring-1 ring-inset",
        tone === "error"
          ? "bg-red-50 text-red-700 ring-red-600/20"
          : "bg-amber-50 text-amber-800 ring-amber-500/20",
      )}
    >
      <Icon icon={Alert02Icon} size={16} className="mt-0.5 shrink-0" />
      <ul className="flex flex-col gap-1">
        {issues.map((issue) => (
          <li key={`${issue.code}-${issue.message}`} className="text-sm">
            {issue.message}
          </li>
        ))}
      </ul>
    </div>
  );
}
