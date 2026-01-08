import { usePostV1CreateNewGraph } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { Graph } from "@/app/api/__generated__/models/graph";
import { GraphModel } from "@/app/api/__generated__/models/graphModel";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { sanitizeImportedGraph } from "@/lib/autogpt-server-api";
import { zodResolver } from "@hookform/resolvers/zod";
import { useEffect, useRef, useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { uploadAgentFormSchema } from "./LibraryUploadAgentDialog";

export function useLibraryUploadAgentDialog() {
  const [isOpen, setIsOpen] = useState(false);
  const { toast } = useToast();
  const [agentObject, setAgentObject] = useState<Graph | null>(null);

  const { mutateAsync: createGraph, isPending: isUploading } =
    usePostV1CreateNewGraph({
      mutation: {
        onSuccess: ({ data }) => {
          setIsOpen(false);
          toast({
            title: "Success",
            description: "Agent uploaded successfully",
            variant: "default",
          });
          const qID = "flowID";
          window.location.href = `/build?${qID}=${(data as GraphModel).id}`;
        },
        onError: () => {
          toast({
            title: "Error",
            description: "Error Uploading agent",
            variant: "destructive",
          });
        },
      },
    });

  const form = useForm<z.infer<typeof uploadAgentFormSchema>>({
    resolver: zodResolver(uploadAgentFormSchema),
    defaultValues: {
      agentName: "",
      agentDescription: "",
      agentFile: "",
    },
  });

  const agentFileValue = form.watch("agentFile");
  const prevAgentObjectRef = useRef<Graph | null>(null);

  useEffect(() => {
    if (!agentFileValue) {
      const prevAgent = prevAgentObjectRef.current;
      if (prevAgent) {
        const currentName = form.getValues("agentName");
        const currentDescription = form.getValues("agentDescription");
        if (currentName === prevAgent.name) {
          form.setValue("agentName", "");
        }
        if (currentDescription === prevAgent.description) {
          form.setValue("agentDescription", "");
        }
      }
      setAgentObject(null);
      prevAgentObjectRef.current = null;
      return;
    }

    try {
      const base64Match = agentFileValue.match(/^data:[^;]+;base64,(.+)$/);
      if (!base64Match) {
        throw new Error("Invalid base64 data URL format");
      }

      const base64String = base64Match[1];
      const jsonString = atob(base64String);
      const obj = JSON.parse(jsonString);

      if (
        !["name", "description", "nodes", "links"].every(
          (key) => key in obj && obj[key] != null,
        )
      ) {
        throw new Error(
          "Invalid agent file. Please upload a valid agent.json file that has been previously exported from the AutoGPT platform. The file must contain the required fields: name, description, nodes, and links.",
        );
      }

      const agent = obj as Graph;
      sanitizeImportedGraph(agent);
      setAgentObject(agent);
      prevAgentObjectRef.current = agent;

      if (!form.getValues("agentName")) {
        form.setValue("agentName", agent.name);
      }
      if (!form.getValues("agentDescription")) {
        form.setValue("agentDescription", agent.description);
      }
    } catch (error) {
      console.error("Error loading agent file:", error);

      toast({
        title: "Invalid Agent File",
        description:
          "Please upload a valid agent.json file that has been previously exported from the AutoGPT platform. The file must contain the required fields: name, description, nodes, and links.",
        duration: 5000,
        variant: "destructive",
      });

      form.resetField("agentFile");
      setAgentObject(null);
    }
  }, [agentFileValue, form, toast]);

  const onSubmit = async (values: z.infer<typeof uploadAgentFormSchema>) => {
    if (!agentObject) {
      form.setError("root", { message: "No Agent object to save" });
      return;
    }

    const payload: Graph = {
      ...agentObject,
      name: values.agentName,
      description: values.agentDescription,
      is_active: true,
    };

    await createGraph({
      data: {
        graph: payload,
        source: "upload",
      },
    });
  };

  return {
    onSubmit,
    isUploading,
    isOpen,
    setIsOpen,
    form,
    agentObject,
  };
}
