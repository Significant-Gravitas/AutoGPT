import { z } from "zod";
import { cn } from "@/lib/utils";
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
} from "@/components/__legacy__/ui/form";
import { Input } from "@/components/__legacy__/ui/input";
import { Button } from "@/components/__legacy__/ui/button";
import { Textarea } from "@/components/__legacy__/ui/textarea";
import { EnterIcon } from "@radix-ui/react-icons";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import {
  Graph,
  GraphCreatable,
  sanitizeImportedGraph,
} from "@/lib/autogpt-server-api";

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

export const AgentImportForm: React.FC<
  React.FormHTMLAttributes<HTMLFormElement>
> = ({ className, ...props }) => {
  const [agentObject, setAgentObject] = useState<GraphCreatable | null>(null);
  const api = useBackendAPI();

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
    };

    api
      .createGraph(payload)
      .then((response) => {
        const qID = "flowID";
        window.location.href = `/build?${qID}=${response.id}`;
      })
      .catch((error) => {
        const entity_type = "agent";
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
                  data-testid="import-agent-file-input"
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
                              (key) => key in obj && obj[key] != null,
                            )
                          ) {
                            throw new Error(
                              "Invalid agent object in file: " +
                                JSON.stringify(obj, null, 2),
                            );
                          }
                          const graph = obj as Graph;
                          sanitizeImportedGraph(graph);
                          setAgentObject(graph);
                          form.setValue("agentName", graph.name);
                          form.setValue("agentDescription", graph.description);
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
                <Input {...field} data-testid="agent-name-input" />
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
                <Textarea {...field} data-testid="agent-description-input" />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <Button
          type="submit"
          className="w-full"
          disabled={!agentObject}
          data-testid="import-agent-submit"
        >
          <EnterIcon className="mr-2" /> Import & Edit
        </Button>
      </form>
    </Form>
  );
};
