"use client";

import React, { useState, useEffect, useMemo } from "react";
import { useRouter } from "next/navigation";
import { useForm, Controller } from "react-hook-form";
import MarketplaceAPI from "@/lib/marketplace-api";
import AutoGPTServerAPI from "@/lib/autogpt-server-api";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import { Checkbox } from "@/components/ui/checkbox";
import {
  MultiSelector,
  MultiSelectorContent,
  MultiSelectorInput,
  MultiSelectorItem,
  MultiSelectorList,
  MultiSelectorTrigger,
} from "@/components/ui/multiselect";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

type FormData = {
  name: string;
  description: string;
  author: string;
  keywords: string[];
  categories: string[];
  agreeToTerms: boolean;
  selectedAgentId: string;
};

const keywords = [
  "Automation",
  "AI Workflows",
  "Integration",
  "Task Automation",
  "Data Processing",
  "Workflow Management",
  "Real-time Analytics",
  "Custom Triggers",
  "Event-driven",
  "API Integration",
  "Data Transformation",
  "Multi-step Workflows",
  "Collaboration Tools",
  "Business Process Automation",
  "No-code Solutions",
  "AI-Powered",
  "Smart Notifications",
  "Data Syncing",
  "User Engagement",
  "Reporting Automation",
  "Lead Generation",
  "Customer Support Automation",
  "E-commerce Automation",
  "Social Media Management",
  "Email Marketing Automation",
  "Document Management",
  "Data Enrichment",
  "Performance Tracking",
  "Predictive Analytics",
  "Resource Allocation",
  "Chatbot",
  "Virtual Assistant",
  "Workflow Automation",
  "Social Media Manager",
  "Email Optimizer",
  "Content Generator",
  "Data Analyzer",
  "Task Scheduler",
  "Customer Service Bot",
  "Personalization Engine",
];

const SubmitPage: React.FC = () => {
  const router = useRouter();
  const {
    control,
    handleSubmit,
    watch,
    setValue,
    formState: { errors },
  } = useForm<FormData>({
    defaultValues: {
      selectedAgentId: "", // Initialize with an empty string
      name: "",
      description: "",
      author: "",
      keywords: [],
      categories: [],
      agreeToTerms: false,
    },
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [userAgents, setUserAgents] = useState<
    Array<{ id: string; name: string; version: number }>
  >([]);
  const [selectedAgentGraph, setSelectedAgentGraph] = useState<any>(null);

  const selectedAgentId = watch("selectedAgentId");

  useEffect(() => {
    const fetchUserAgents = async () => {
      const api = new AutoGPTServerAPI();
      const agents = await api.listGraphs();
      console.log(agents);
      setUserAgents(
        agents.map((agent) => ({
          id: agent.id,
          name: agent.name || `Agent (${agent.id})`,
          version: agent.version,
        })),
      );
    };

    fetchUserAgents();
  }, []);

  useEffect(() => {
    const fetchAgentGraph = async () => {
      if (selectedAgentId) {
        const api = new AutoGPTServerAPI();
        const graph = await api.getGraph(selectedAgentId, undefined, true);
        setSelectedAgentGraph(graph);
        setValue("name", graph.name);
        setValue("description", graph.description);
      }
    };

    fetchAgentGraph();
  }, [selectedAgentId, setValue]);

  const onSubmit = async (data: FormData) => {
    setIsSubmitting(true);
    setSubmitError(null);

    if (!data.agreeToTerms) {
      throw new Error("You must agree to the terms of service");
    }

    try {
      if (!selectedAgentGraph) {
        throw new Error("Please select an agent");
      }

      const api = new MarketplaceAPI();
      await api.submitAgent(
        {
          ...selectedAgentGraph,
          name: data.name,
          description: data.description,
        },
        data.author,
        data.keywords,
        data.categories,
      );

      router.push("/marketplace?submission=success");
    } catch (error) {
      console.error("Submission error:", error);
      setSubmitError(
        error instanceof Error ? error.message : "An unknown error occurred",
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="mb-6 text-3xl font-bold">Submit Your Agent</h1>
      <Card className="p-6">
        <form onSubmit={handleSubmit(onSubmit)}>
          <div className="space-y-4">
            <Controller
              name="selectedAgentId"
              control={control}
              rules={{ required: "Please select an agent" }}
              render={({ field }) => (
                <div>
                  <label
                    htmlFor={field.name}
                    className="block text-sm font-medium text-gray-700"
                  >
                    Select Agent
                  </label>
                  <Select
                    onValueChange={field.onChange}
                    value={field.value || ""}
                  >
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder="Select an agent" />
                    </SelectTrigger>
                    <SelectContent>
                      {userAgents.map((agent) => (
                        <SelectItem key={agent.id} value={agent.id}>
                          {agent.name} (v{agent.version})
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  {errors.selectedAgentId && (
                    <p className="mt-1 text-sm text-red-600">
                      {errors.selectedAgentId.message}
                    </p>
                  )}
                </div>
              )}
            />

            {/* {selectedAgentGraph && (
              <div className="mt-4" style={{ height: "600px" }}>
                <ReactFlow
                  nodes={nodes}
                  edges={edges}
                  fitView
                  attributionPosition="bottom-left"
                  nodesConnectable={false}
                  nodesDraggable={false}
                  zoomOnScroll={false}
                  panOnScroll={false}
                  elementsSelectable={false}
                >
                  <Controls showInteractive={false} />
                  <Background />
                </ReactFlow>
              </div>
            )} */}

            <Controller
              name="name"
              control={control}
              rules={{ required: "Name is required" }}
              render={({ field }) => (
                <div>
                  <label
                    htmlFor={field.name}
                    className="block text-sm font-medium text-gray-700"
                  >
                    Agent Name
                  </label>
                  <Input
                    id={field.name}
                    placeholder="Enter your agent's name"
                    {...field}
                  />
                  {errors.name && (
                    <p className="mt-1 text-sm text-red-600">
                      {errors.name.message}
                    </p>
                  )}
                </div>
              )}
            />

            <Controller
              name="description"
              control={control}
              rules={{ required: "Description is required" }}
              render={({ field }) => (
                <div>
                  <label
                    htmlFor={field.name}
                    className="block text-sm font-medium text-gray-700"
                  >
                    Description
                  </label>
                  <Textarea
                    id={field.name}
                    placeholder="Describe your agent"
                    {...field}
                  />
                  {errors.description && (
                    <p className="mt-1 text-sm text-red-600">
                      {errors.description.message}
                    </p>
                  )}
                </div>
              )}
            />

            <Controller
              name="author"
              control={control}
              rules={{ required: "Author is required" }}
              render={({ field }) => (
                <div>
                  <label
                    htmlFor={field.name}
                    className="block text-sm font-medium text-gray-700"
                  >
                    Author
                  </label>
                  <Input
                    id={field.name}
                    placeholder="Your name or username"
                    {...field}
                  />
                  {errors.author && (
                    <p className="mt-1 text-sm text-red-600">
                      {errors.author.message}
                    </p>
                  )}
                </div>
              )}
            />

            <Controller
              name="keywords"
              control={control}
              rules={{ required: "At least one keyword is required" }}
              render={({ field }) => (
                <div>
                  <label
                    htmlFor={field.name}
                    className="block text-sm font-medium text-gray-700"
                  >
                    Keywords
                  </label>
                  <MultiSelector
                    values={field.value || []}
                    onValuesChange={field.onChange}
                  >
                    <MultiSelectorTrigger>
                      <MultiSelectorInput placeholder="Add keywords" />
                    </MultiSelectorTrigger>
                    <MultiSelectorContent>
                      <MultiSelectorList>
                        {keywords.map((keyword) => (
                          <MultiSelectorItem key={keyword} value={keyword}>
                            {keyword}
                          </MultiSelectorItem>
                        ))}
                        {/* Add more predefined keywords as needed */}
                      </MultiSelectorList>
                    </MultiSelectorContent>
                  </MultiSelector>
                  {errors.keywords && (
                    <p className="mt-1 text-sm text-red-600">
                      {errors.keywords.message}
                    </p>
                  )}
                </div>
              )}
            />

            <Controller
              name="categories"
              control={control}
              rules={{ required: "At least one category is required" }}
              render={({ field }) => (
                <div>
                  <label
                    htmlFor={field.name}
                    className="block text-sm font-medium text-gray-700"
                  >
                    Categories
                  </label>
                  <MultiSelector
                    values={field.value || []}
                    onValuesChange={field.onChange}
                  >
                    <MultiSelectorTrigger>
                      <MultiSelectorInput placeholder="Select categories" />
                    </MultiSelectorTrigger>
                    <MultiSelectorContent>
                      <MultiSelectorList>
                        <MultiSelectorItem value="productivity">
                          Productivity
                        </MultiSelectorItem>
                        <MultiSelectorItem value="entertainment">
                          Entertainment
                        </MultiSelectorItem>
                        <MultiSelectorItem value="education">
                          Education
                        </MultiSelectorItem>
                        <MultiSelectorItem value="business">
                          Business
                        </MultiSelectorItem>
                        <MultiSelectorItem value="other">
                          Other
                        </MultiSelectorItem>
                      </MultiSelectorList>
                    </MultiSelectorContent>
                  </MultiSelector>
                  {errors.categories && (
                    <p className="mt-1 text-sm text-red-600">
                      {errors.categories.message}
                    </p>
                  )}
                </div>
              )}
            />

            <Controller
              name="agreeToTerms"
              control={control}
              rules={{ required: "You must agree to the terms of service" }}
              render={({ field }) => (
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="agreeToTerms"
                    checked={field.value}
                    onCheckedChange={field.onChange}
                  />
                  <label
                    htmlFor="agreeToTerms"
                    className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                  >
                    I agree to the{" "}
                    <a href="/terms" className="text-blue-500 hover:underline">
                      terms of service
                    </a>
                  </label>
                </div>
              )}
            />
            {errors.agreeToTerms && (
              <p className="mt-1 text-sm text-red-600">
                {errors.agreeToTerms.message}
              </p>
            )}

            {submitError && (
              <Alert variant="destructive">
                <AlertTitle>Submission Failed</AlertTitle>
                <AlertDescription>{submitError}</AlertDescription>
              </Alert>
            )}

            <Button type="submit" className="w-full" disabled={isSubmitting}>
              {isSubmitting ? "Submitting..." : "Submit Agent"}
            </Button>
          </div>
        </form>
      </Card>
    </div>
  );
};

export default SubmitPage;
