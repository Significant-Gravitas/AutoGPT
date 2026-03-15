import { useCallback, useEffect } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { parseAsInteger, parseAsString, useQueryStates } from "nuqs";
import { useGetV1GetSpecificGraph } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { GraphModel } from "@/app/api/__generated__/models/graphModel";
import { useControlPanelStore } from "../../../stores/controlPanelStore";
import { useSaveGraph } from "../../../hooks/useSaveGraph";

const formSchema = z.object({
  name: z.string().min(1, "Name is required").max(100),
  description: z.string().max(500),
});

type SaveableGraphFormValues = z.infer<typeof formSchema>;

export const useNewSaveControl = () => {
  const { setSaveControlOpen } = useControlPanelStore();

  const onSuccess = (graph: GraphModel) => {
    setSaveControlOpen(false);
    form.reset({
      name: graph.name,
      description: graph.description,
    });
  };

  const { saveGraph, isSaving } = useSaveGraph({
    showToast: true,
    onSuccess,
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
        enabled: !!flowID,
      },
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
      saveGraph(values);
    },
    [saveGraph],
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

  return {
    form,
    isSaving: isSaving,
    graphVersion: graph?.version,
    handleSave,
  };
};
