"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";

import { usePostV2CreateWorkspace } from "@/app/api/__generated__/endpoints/orgs/orgs";
import type { TeamResponse } from "@/app/api/__generated__/models/teamResponse";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useOrgTeamStore } from "@/services/org-team/store";

import { createTeamSchema, type CreateTeamFormValues } from "./schema";

interface Args {
  orgId: string;
  onCreated: () => void;
  onClose: () => void;
}

export function useCreateTeamDialog({ orgId, onCreated, onClose }: Args) {
  const { teams, setTeams } = useOrgTeamStore();

  const form = useForm<CreateTeamFormValues>({
    resolver: zodResolver(createTeamSchema),
    defaultValues: { name: "", description: "", join_policy: "OPEN" },
    mode: "onChange",
  });

  const { mutateAsync: createTeam, isPending } = usePostV2CreateWorkspace({
    mutation: {
      onError: (error) => {
        toast({
          title: "Failed to create team",
          description:
            error instanceof Error ? error.message : "Please try again.",
          variant: "destructive",
        });
      },
    },
  });

  async function handleSubmit(values: CreateTeamFormValues) {
    const response = await createTeam({
      orgId,
      data: {
        name: values.name,
        description: values.description || null,
        join_policy: values.join_policy,
      },
    });
    const team = response.data as TeamResponse;
    setTeams([
      ...teams,
      {
        id: team.id,
        name: team.name,
        slug: team.slug,
        isDefault: team.is_default,
        joinPolicy: team.join_policy,
        orgId: team.org_id,
      },
    ]);
    toast({ title: `Team "${team.name}" created`, variant: "success" });
    onCreated();
    handleClose();
  }

  function handleClose() {
    form.reset();
    onClose();
  }

  return {
    form,
    isPending,
    handleSubmit,
    handleClose,
  };
}
