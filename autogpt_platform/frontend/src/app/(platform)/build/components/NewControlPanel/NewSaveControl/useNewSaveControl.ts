import { useEffect } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { parseAsInteger, parseAsString, useQueryStates } from "nuqs";
import {
  getGetV1GetSpecificGraphQueryKey,
  useGetV1GetSpecificGraph,
  usePostV1CreateNewGraph,
  usePutV1UpdateGraphVersion,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import { GraphModel } from "@/app/api/__generated__/models/graphModel";
import { useNodeStore } from "../../../stores/nodeStore";
import { useEdgeStore } from "../../../stores/edgeStore";
import { Graph } from "@/app/api/__generated__/models/graph";
import { useControlPanelStore } from "../../../stores/controlPanelStore";
import { graphsEquivalent } from "./helpers";

const formSchema = z.object({
  name: z.string().min(1, "Name is required").max(100),
  description: z.string().max(500),
});

type SaveableGraphFormValues = z.infer<typeof formSchema>;

export const useNewSaveControl = () => {
  const { setSaveControlOpen } = useControlPanelStore();
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const [{ flowID, flowVersion }, setQueryStates] = useQueryStates({
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

  const { mutateAsync: createNewGraph, isPending: isCreating } =
    usePostV1CreateNewGraph({
      mutation: {
        onSuccess: (response) => {
          const data = response.data as GraphModel;
          form.reset({
            name: data.name,
            description: data.description,
          });
          setSaveControlOpen(false);
          setQueryStates({
            flowID: data.id,
            flowVersion: data.version,
          });
          toast({
            title: "All changes saved successfully!",
          });
        },
        onError: (error) => {
          toast({
            title: (error.detail as string) ?? "An unexpected error occurred.",
            description: "An unexpected error occurred.",
            variant: "destructive",
          });
        },
      },
    });

  const { mutateAsync: updateGraph, isPending: isUpdating } =
    usePutV1UpdateGraphVersion({
      mutation: {
        onSuccess: (response) => {
          const data = response.data as GraphModel;
          form.reset({
            name: data.name,
            description: data.description,
          });
          setSaveControlOpen(false);
          setQueryStates({
            flowID: data.id,
            flowVersion: data.version,
          });
          toast({
            title: "All changes saved successfully!",
          });
          queryClient.invalidateQueries({
            queryKey: getGetV1GetSpecificGraphQueryKey(data.id),
          });
        },
        onError: (error) => {
          toast({
            title: (error.detail as string) ?? "An unexpected error occurred.",
            description: "An unexpected error occurred.",
            variant: "destructive",
          });
        },
      },
    });

  const form = useForm<SaveableGraphFormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      name: graph?.name ?? "",
      description: graph?.description ?? "",
    },
  });

  // Handle Ctrl+S / Cmd+S keyboard shortcut
  useEffect(() => {
    const handleKeyDown = async (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "s") {
        event.preventDefault();
        await onSubmit(form.getValues());
      }
    };

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [form]);

  useEffect(() => {
    if (graph) {
      form.reset({
        name: graph.name ?? "",
        description: graph.description ?? "",
      });
    }
  }, [graph, form]);

  const onSubmit = async (values: SaveableGraphFormValues) => {
    const graphNodes = useNodeStore.getState().getBackendNodes();
    const graphLinks = useEdgeStore.getState().getBackendLinks();

    if (graph && graph.id) {
      const data: Graph = {
        id: graph.id,
        name: values.name,
        description: values.description,
        nodes: graphNodes,
        links: graphLinks,
      };
      if (graphsEquivalent(graph, data)) {
        toast({
          title: "No changes to save",
          description: "The graph is the same as the saved version.",
          variant: "default",
        });
        return;
      }
      await updateGraph({ graphId: graph.id, data: data });
    } else {
      const data: Graph = {
        name: values.name,
        description: values.description,
        nodes: graphNodes,
        links: graphLinks,
      };
      await createNewGraph({ data: { graph: data } });
    }
  };

  return {
    form,
    isLoading: isCreating || isUpdating,
    graphVersion: graph?.version,
    onSubmit,
  };
};
