"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";

import { usePatchV2UpdateWorkspace } from "@/app/api/__generated__/endpoints/orgs/orgs";
import type { TeamResponse } from "@/app/api/__generated__/models/teamResponse";
import { toast } from "@/components/molecules/Toast/use-toast";
import { TEAM_HEADER_NAME } from "@/services/org-team/headers";

const teamProfileSchema = z.object({
  name: z
    .string()
    .trim()
    .min(1, "Name is required")
    .max(100, "Name must be 100 characters or less"),
  description: z
    .string()
    .trim()
    .max(500, "Description must be 500 characters or less")
    .optional(),
  join_policy: z.enum(["OPEN", "PRIVATE"]),
});

export type TeamProfileFormValues = z.infer<typeof teamProfileSchema>;

interface Args {
  orgId: string;
  team: TeamResponse;
  onSaved: () => void;
}

export function useTeamProfileForm({ orgId, team, onSaved }: Args) {
  const form = useForm<TeamProfileFormValues>({
    resolver: zodResolver(teamProfileSchema),
    defaultValues: {
      name: team.name,
      description: team.description ?? "",
      join_policy: team.join_policy === "PRIVATE" ? "PRIVATE" : "OPEN",
    },
    mode: "onChange",
  });

  const { mutateAsync: updateTeam, isPending } = usePatchV2UpdateWorkspace({
    mutation: {
      onError: (error) => {
        toast({
          title: "Failed to update team",
          description:
            error instanceof Error ? error.message : "Please try again.",
          variant: "destructive",
        });
      },
    },
    request: { headers: { [TEAM_HEADER_NAME]: team.id } },
  });

  async function handleSubmit(values: TeamProfileFormValues) {
    const response = await updateTeam({
      orgId,
      wsId: team.id,
      data: {
        name: values.name !== team.name ? values.name : undefined,
        description:
          (values.description || "") !== (team.description ?? "")
            ? values.description || null
            : undefined,
        join_policy:
          values.join_policy !== team.join_policy
            ? values.join_policy
            : undefined,
      },
    });
    const updated = response.data as TeamResponse;
    form.reset({
      name: updated.name,
      description: updated.description ?? "",
      join_policy: updated.join_policy === "PRIVATE" ? "PRIVATE" : "OPEN",
    });
    toast({ title: "Team updated", variant: "success" });
    onSaved();
  }

  return {
    form,
    isPending,
    isDirty: form.formState.isDirty,
    handleSubmit,
  };
}
