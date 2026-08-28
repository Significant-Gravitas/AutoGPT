import { useCallback, useEffect } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { parseAsInteger, parseAsString, useQueryStates } from "nuqs";
import {
  getGetV1GetSpecificGraphQueryKey,
  useGetV1GetSpecificGraph,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import { GraphModel } from "@/app/api/__generated__/models/graphModel";
import { useControlPanelStore } from "../../../stores/controlPanelStore";
import { useSaveGraph } from "../../../hooks/useSaveGraph";
import { useCreateTeamSelection } from "@/components/contextual/TeamPicker/useCreateTeamSelection";
import {
  CreateSurface,
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";
import { useBuilderTenantScope } from "../../../hooks/useBuilderTenantScope";

const formSchema = z.object({
  name: z.string().min(1, "Name is required").max(100),
  description: z.string().max(500),
});

type SaveableGraphFormValues = z.infer<typeof formSchema>;

export const useNewSaveControl = () => {
  const tenantScope = useBuilderTenantScope();
  const { setSaveControlOpen } = useControlPanelStore();

  const onSuccess = (graph: GraphModel) => {
    setSaveControlOpen(false);
    form.reset({
      name: graph.name,
      description: graph.description,
    });
  };

  // Only a brand-new agent gets a team owner; saving over an existing graph
  // (flowID present) reuses its ownership, so hide the picker then.
  const { teamId, setTeamId, hasTeams, isReady } = useCreateTeamSelection(
    CreateSurface.BuilderSave,
  );

  const { saveGraph, isSaving } = useSaveGraph({
    showToast: true,
    onSuccess,
    teamId,
  });

  const [{ flowID, flowVersion }] = useQueryStates({
    flowID: parseAsString,
    flowVersion: parseAsInteger,
  });

  const { data: graph } = useGetV1GetSpecificGraph(
    flowID ?? "",
    flowVersion !== null ? { version: flowVersion } : {},
    {
      query: {
        select: (res) => res.data as GraphModel,
        enabled: !!flowID && tenantScope.isReady,
        queryKey: getTeamScopedQueryKey(
          getGetV1GetSpecificGraphQueryKey(
            flowID ?? "",
            flowVersion !== null ? { version: flowVersion } : {},
          ),
          tenantScope.organizationId,
          tenantScope.teamId,
        ),
      },
      request: getTenantRequestInit(
        tenantScope.organizationId,
        tenantScope.teamId,
        tenantScope.isReady,
      ),
    },
  );

  const form = useForm<SaveableGraphFormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      name: graph?.name ?? "",
      description: graph?.description ?? "",
    },
  });

  const handleSave = useCallback(
    (values: SaveableGraphFormValues) => {
      if (!isReady) return;
      saveGraph(values);
    },
    [isReady, saveGraph],
  );

  useEffect(() => {
    const handleKeyDown = async (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "s") {
        event.preventDefault();
        handleSave(form.getValues());
      }
    };

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [handleSave]);

  useEffect(() => {
    if (graph) {
      form.reset({
        name: graph.name ?? "",
        description: graph.description ?? "",
      });
    }
  }, [graph, form]);

  // Show the picker only when creating a new agent (no flowID) and the user
  // actually has teams to choose from.
  const showTeamPicker = !flowID && hasTeams;

  return {
    form,
    isSaving: isSaving,
    graphVersion: graph?.version,
    handleSave,
    teamId,
    setTeamId,
    showTeamPicker,
    isReady,
  };
};
