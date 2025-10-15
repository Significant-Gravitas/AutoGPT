import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { uploadAgentFormSchema } from "./LibraryUploadAgentDialog";
import { usePostV1CreateNewGraph } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { GraphModel } from "@/app/api/__generated__/models/graphModel";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useState } from "react";
import { Graph } from "@/app/api/__generated__/models/graph";
import { sanitizeImportedGraph } from "@/lib/autogpt-server-api";

export const useLibraryUploadAgentDialog = () => {
  const [isDroped, setisDroped] = useState(false);
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
    },
  });

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
      },
    });
  };

  const handleChange = (file: File) => {
    setTimeout(() => {
      setisDroped(false);
    }, 2000);

    form.setValue("agentFile", file);
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const obj = JSON.parse(event.target?.result as string);
        if (
          !["name", "description", "nodes", "links"].every(
            (key) => key in obj && obj[key] != null,
          )
        ) {
          throw new Error(
            "Invalid agent object in file: " + JSON.stringify(obj, null, 2),
          );
        }
        const agent = obj as Graph;

        // Validate Input/Output nodes have required 'name' field
        // These are the IO block IDs that require a 'name' in input_default
        const IO_BLOCK_IDS = [
          "c0a8e994-ebf1-4a9c-a4d8-89d09c86741b", // AgentInputBlock
          "363ae599-353e-4804-937e-b2ee3cef3da4", // AgentOutputBlock
          "7fcd3bcb-8e1b-4e69-903d-32d3d4a92158", // AgentShortTextInputBlock
          "90a56ffb-7024-4b2b-ab50-e26c5e5ab8ba", // AgentLongTextInputBlock
          "96dae2bb-97a2-41c2-bd2f-13a3b5a8ea98", // AgentNumberInputBlock
          "7e198b09-4994-47db-8b4d-952d98241817", // AgentDateInputBlock
          "2a1c757e-86cf-4c7e-aacf-060dc382e434", // AgentTimeInputBlock
          "95ead23f-8283-4654-aef3-10c053b74a31", // AgentFileInputBlock
          "655d6fdf-a334-421c-b733-520549c07cd1", // AgentDropdownInputBlock
          "cbf36ab5-df4a-43b6-8a7f-f7ed8652116e", // AgentToggleInputBlock
          "5603b273-f41e-4020-af7d-fbc9c6a8d928", // AgentTableInputBlock
        ];

        for (const node of agent.nodes || []) {
          if (IO_BLOCK_IDS.includes(node.block_id)) {
            if (
              !node.input_default ||
              typeof node.input_default !== "object" ||
              !("name" in node.input_default) ||
              !node.input_default.name
            ) {
              throw new Error(
                `Invalid Input/Output node (ID: ${node.id}): missing required 'name' field in input_default. ` +
                  `All Input and Output blocks must have a 'name' field.`,
              );
            }
          }
        }

        sanitizeImportedGraph(agent);
        setAgentObject(agent);
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
            error instanceof Error
              ? error.message
              : "The uploaded file is not a valid agent configuration.",
          variant: "destructive",
        });
        form.setValue("agentFile", undefined as any);
      }
    };
    reader.readAsText(file);
    setisDroped(false);
  };

  const clearAgentFile = () => {
    const currentName = form.getValues("agentName");
    const currentDescription = form.getValues("agentDescription");
    const prevAgent = agentObject;

    form.setValue("agentFile", undefined as any);
    if (prevAgent && currentName === prevAgent.name) {
      form.setValue("agentName", "");
    }
    if (prevAgent && currentDescription === prevAgent.description) {
      form.setValue("agentDescription", "");
    }

    setAgentObject(null);
  };

  return {
    onSubmit,
    isUploading,
    isOpen,
    setIsOpen,
    form,
    agentObject,
    isDroped,
    handleChange,
    setisDroped,
    clearAgentFile,
  };
};
