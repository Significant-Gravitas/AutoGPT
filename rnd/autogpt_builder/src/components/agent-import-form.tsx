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

const formSchema = z.object({
  agentFile: z.instanceof(File),
  agentName: z.string().min(1, "Agent name is required"),
  agentDescription: z.string(),
  importAsTemplate: z.boolean(),
});

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
                <div className="flex space-x-2 items-center">
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
