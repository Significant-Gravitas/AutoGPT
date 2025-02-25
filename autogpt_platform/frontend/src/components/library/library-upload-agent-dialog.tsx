"use client";
import { useState } from "react";
import { Upload, X } from "lucide-react";
import { removeCredentials } from "@/lib/utils";
import { Button } from "@/components/agptui/Button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { z } from "zod";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { FileUploader } from "react-drag-drop-files";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Graph, GraphCreatable } from "@/lib/autogpt-server-api";
import { updatedBlockIDMap } from "@/components/agent-import-form";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { useToast } from "@/components/ui/use-toast";

const fileTypes = ["JSON"];

const fileSchema = z.custom<File>((val) => val instanceof File, {
  message: "Must be a File object",
});

const formSchema = z.object({
  agentFile: fileSchema,
  agentName: z.string().min(1, "Agent name is required"),
  agentDescription: z.string(),
});

function updateBlockIDs(graph: Graph) {
  graph.nodes
    .filter((node) => node.block_id in updatedBlockIDMap)
    .forEach((node) => {
      node.block_id = updatedBlockIDMap[node.block_id];
    });
  return graph;
}

export default function LibraryUploadAgentDialog(): React.ReactNode {
  const [isDroped, setisDroped] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const api = useBackendAPI();
  const { toast } = useToast();
  const [agentObject, setAgentObject] = useState<GraphCreatable | null>(null);

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      agentName: "",
      agentDescription: "",
    },
  });

  const onSubmit = async (values: z.infer<typeof formSchema>) => {
    if (!agentObject) {
      form.setError("root", { message: "No Agent object to save" });
      return;
    }

    setIsLoading(true);

    const payload: GraphCreatable = {
      ...agentObject,
      name: values.agentName,
      description: values.agentDescription,
      is_active: true,
    };

    try {
      const response = await api.createGraph(payload);
      setIsOpen(false);
      toast({
        title: "Success",
        description: "Agent uploaded successfully",
        variant: "default",
      });
      const qID = "flowID";
      window.location.href = `/build?${qID}=${response.id}`;
    } catch (error) {
      form.setError("root", {
        message: `Could not create agent: ${error}`,
      });
    } finally {
      setIsLoading(false);
    }
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
        removeCredentials(agent);
        updateBlockIDs(agent);
        setAgentObject(agent);
        if (!form.getValues("agentName")) {
          form.setValue("agentName", agent.name);
        }
        if (!form.getValues("agentDescription")) {
          form.setValue("agentDescription", agent.description);
        }
      } catch (error) {
        console.error("Error loading agent file:", error);
      }
    };
    reader.readAsText(file);
    setisDroped(false);
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="primary" className="w-fit sm:w-[177px]">
          <Upload className="h-5 w-5 sm:mr-2" />
          <span className="hidden items-center sm:inline-flex">
            Upload an agent
          </span>
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="mb-8 text-center">Upload Agent</DialogTitle>
        </DialogHeader>

        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            <FormField
              control={form.control}
              name="agentName"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Agent name</FormLabel>
                  <FormControl>
                    <Input {...field} className="w-full rounded-[10px]" />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="agentDescription"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Description</FormLabel>
                  <FormControl>
                    <Textarea {...field} className="w-full rounded-[10px]" />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="agentFile"
              render={({ field }) => (
                <FormItem className="rounded-xl border-2 border-dashed border-neutral-300 hover:border-neutral-600">
                  <FormControl>
                    {field.value ? (
                      <div className="relative flex rounded-[10px] border p-2 font-sans text-sm font-medium text-[#525252] outline-none">
                        <span className="line-clamp-1">{field.value.name}</span>
                        <Button
                          onClick={() =>
                            form.setValue("agentFile", undefined as any)
                          }
                          className="absolute left-[-10px] top-[-16px] mt-2 h-fit border-none bg-red-200 p-1"
                        >
                          <X
                            className="m-0 h-[12px] w-[12px] text-red-600"
                            strokeWidth={3}
                          />
                        </Button>
                      </div>
                    ) : (
                      <FileUploader
                        handleChange={handleChange}
                        name="file"
                        types={fileTypes}
                        label={"Upload your agent here..!!"}
                        uploadedLabel={"Uploading Successful"}
                        required={true}
                        hoverTitle={"Drop your agent here...!!"}
                        maxSize={10}
                        classes={"drop-style"}
                        onDrop={() => {
                          setisDroped(true);
                        }}
                        onSelect={() => setisDroped(true)}
                      >
                        <div
                          style={{
                            minHeight: "150px",
                            display: "flex",
                            flexDirection: "column",
                            justifyContent: "center",
                            alignItems: "center",
                            outline: "none",
                            fontFamily: "var(--font-geist-sans)",
                            color: "#525252",
                            fontSize: "14px",
                            fontWeight: "500",
                            borderWidth: "0px",
                          }}
                        >
                          {isDroped ? (
                            <div className="flex items-center justify-center py-4">
                              <div className="h-8 w-8 animate-spin rounded-full border-b-2 border-t-2 border-neutral-800"></div>
                            </div>
                          ) : (
                            <>
                              <span>Drop your agent here</span>
                              <span>or</span>
                              <span>Click to upload</span>
                            </>
                          )}
                        </div>
                      </FileUploader>
                    )}
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <Button
              type="submit"
              variant="primary"
              className="mt-2 self-end"
              disabled={!agentObject || isLoading}
            >
              {isLoading ? (
                <div className="flex items-center gap-2">
                  <div className="h-4 w-4 animate-spin rounded-full border-b-2 border-t-2 border-white"></div>
                  <span>Uploading...</span>
                </div>
              ) : (
                "Upload Agent"
              )}
            </Button>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
}
