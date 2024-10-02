import { z } from "zod";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import React, { useState } from "react";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import AutoGPTServerAPI, {
  Graph,
  GraphCreatable,
} from "@/lib/autogpt-server-api";
import { cn } from "@/lib/utils";
import { EnterIcon } from "@radix-ui/react-icons";

// Add this custom schema for File type
const fileSchema = z.custom<File>((val) => val instanceof File, {
  message: "Must be a File object",
});

const formSchema = z.object({
  agentFile: fileSchema,
  agentName: z.string().min(1, "Agent name is required"),
  agentDescription: z.string(),
  importAsTemplate: z.boolean(),
});

function updateBlockIDs(graph: Graph) {
  // https://github.com/Significant-Gravitas/AutoGPT/issues/8223
  const updatedBlockIDMap: Record<string, string> = {
    "a1b2c3d4-5e6f-7g8h-9i0j-k1l2m3n4o5p6":
      "436c3984-57fd-4b85-8e9a-459b356883bd",
    "b2g2c3d4-5e6f-7g8h-9i0j-k1l2m3n4o5p6":
      "0e50422c-6dee-4145-83d6-3a5a392f65de",
    "c3d4e5f6-7g8h-9i0j-1k2l-m3n4o5p6q7r8":
      "a0a69be1-4528-491c-a85a-a4ab6873e3f0",
    "c3d4e5f6-g7h8-i9j0-k1l2-m3n4o5p6q7r8":
      "32a87eab-381e-4dd4-bdb8-4c47151be35a",
    "b2c3d4e5-6f7g-8h9i-0j1k-l2m3n4o5p6q7":
      "87840993-2053-44b7-8da4-187ad4ee518c",
    "h1i2j3k4-5l6m-7n8o-9p0q-r1s2t3u4v5w6":
      "d0822ab5-9f8a-44a3-8971-531dd0178b6b",
    "d3f4g5h6-1i2j-3k4l-5m6n-7o8p9q0r1s2t":
      "df06086a-d5ac-4abb-9996-2ad0acb2eff7",
    "h5e7f8g9-1b2c-3d4e-5f6g-7h8i9j0k1l2m":
      "f5b0f5d0-1862-4d61-94be-3ad0fa772760",
    "a1234567-89ab-cdef-0123-456789abcdef":
      "4335878a-394e-4e67-adf2-919877ff49ae",
    "f8e7d6c5-b4a3-2c1d-0e9f-8g7h6i5j4k3l":
      "f66a3543-28d3-4ab5-8945-9b336371e2ce",
    "b29c1b50-5d0e-4d9f-8f9d-1b0e6fcbf0h2":
      "716a67b3-6760-42e7-86dc-18645c6e00fc",
    "31d1064e-7446-4693-o7d4-65e5ca9110d1":
      "cc10ff7b-7753-4ff2-9af6-9399b1a7eddc",
    "c6731acb-4105-4zp1-bc9b-03d0036h370g":
      "5ebe6768-8e5d-41e3-9134-1c7bd89a8d52",
  };
  graph.nodes
    .filter((node) => node.block_id in updatedBlockIDMap)
    .forEach((node) => {
      node.block_id = updatedBlockIDMap[node.block_id];
    });
  return graph;
}

export const AgentImportForm: React.FC<
  React.FormHTMLAttributes<HTMLFormElement>
> = ({ className, ...props }) => {
  const [agentObject, setAgentObject] = useState<GraphCreatable | null>(null);
  const api = new AutoGPTServerAPI();

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      agentName: "",
      agentDescription: "",
      importAsTemplate: false,
    },
  });

  function onSubmit(values: z.infer<typeof formSchema>) {
    if (!agentObject) {
      form.setError("root", { message: "No Agent object to save" });
      return;
    }
    const payload: GraphCreatable = {
      ...agentObject,
      name: values.agentName,
      description: values.agentDescription,
      is_active: !values.importAsTemplate,
      is_template: values.importAsTemplate,
    };

    (values.importAsTemplate
      ? api.createTemplate(payload)
      : api.createGraph(payload)
    )
      .then((response) => {
        const qID = values.importAsTemplate ? "templateID" : "flowID";
        window.location.href = `/build?${qID}=${response.id}`;
      })
      .catch((error) => {
        const entity_type = values.importAsTemplate ? "template" : "agent";
        form.setError("root", {
          message: `Could not create ${entity_type}: ${error}`,
        });
      });
  }

  return (
    <Form {...form}>
      <form
        onSubmit={form.handleSubmit(onSubmit)}
        className={cn("space-y-4", className)}
        {...props}
      >
        <FormField
          control={form.control}
          name="agentFile"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Agent file</FormLabel>
              <FormControl className="cursor-pointer">
                <Input
                  type="file"
                  accept="application/json"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) {
                      field.onChange(file);
                      const reader = new FileReader();
                      // Attach parser to file reader
                      reader.onload = (event) => {
                        try {
                          const obj = JSON.parse(
                            event.target?.result as string,
                          );
                          if (
                            !["name", "description", "nodes", "links"].every(
                              (key) => !!obj[key],
                            )
                          ) {
                            throw new Error(
                              "Invalid agent object in file: " +
                                JSON.stringify(obj, null, 2),
                            );
                          }
                          const agent = obj as Graph;
                          updateBlockIDs(agent);
                          setAgentObject(agent);
                          form.setValue("agentName", agent.name);
                          form.setValue("agentDescription", agent.description);
                          form.setValue("importAsTemplate", agent.is_template);
                        } catch (error) {
                          console.error("Error loading agent file:", error);
                        }
                      };
                      // Load file
                      reader.readAsText(file);
                    }
                  }}
                />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <FormField
          control={form.control}
          name="agentName"
          disabled={!agentObject}
          render={({ field }) => (
            <FormItem>
              <FormLabel>Agent name</FormLabel>
              <FormControl>
                <Input {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <FormField
          control={form.control}
          name="agentDescription"
          disabled={!agentObject}
          render={({ field }) => (
            <FormItem>
              <FormLabel>Agent description</FormLabel>
              <FormControl>
                <Textarea {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <FormField
          control={form.control}
          name="importAsTemplate"
          disabled={!agentObject}
          render={({ field }) => (
            <FormItem>
              <FormLabel>Import as</FormLabel>
              <FormControl>
                <div className="flex items-center space-x-2">
                  <span
                    className={
                      field.value ? "text-gray-400 dark:text-gray-600" : ""
                    }
                  >
                    Agent
                  </span>
                  <Switch
                    disabled={field.disabled}
                    checked={field.value}
                    onCheckedChange={field.onChange}
                  />
                  <span
                    className={
                      field.value ? "" : "text-gray-400 dark:text-gray-600"
                    }
                  >
                    Template
                  </span>
                </div>
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <Button type="submit" className="w-full" disabled={!agentObject}>
          <EnterIcon className="mr-2" /> Import & Edit
        </Button>
      </form>
    </Form>
  );
};
