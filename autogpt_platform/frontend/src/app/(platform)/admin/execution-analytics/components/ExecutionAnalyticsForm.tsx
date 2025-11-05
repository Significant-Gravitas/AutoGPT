"use client";

import { useState } from "react";
import { Button } from "@/components/__legacy__/ui/button";
import { Input } from "@/components/__legacy__/ui/input";
import { Label } from "@/components/__legacy__/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/__legacy__/ui/select";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { usePostV2GenerateExecutionAnalytics } from "@/app/api/__generated__/endpoints/admin/admin";
import { AnalyticsResultsTable } from "./AnalyticsResultsTable";

interface ExecutionAnalyticsRequest {
  graph_id: string;
  graph_version?: number;
  user_id?: string;
  created_after?: string;
  model_name: string;
  batch_size: number;
}

interface ExecutionAnalyticsResult {
  agent_id: string;
  version_id: number;
  user_id: string;
  exec_id: string;
  summary_text?: string;
  score?: number;
  status: "success" | "failed" | "skipped";
  error_message?: string;
}

interface ExecutionAnalyticsResponse {
  total_executions: number;
  processed_executions: number;
  successful_analytics: number;
  failed_analytics: number;
  skipped_executions: number;
  results: ExecutionAnalyticsResult[];
}

const MODEL_OPTIONS = [
  { value: "gpt-4o-mini", label: "GPT-4o Mini (Recommended)" },
  { value: "gpt-4o", label: "GPT-4o" },
  { value: "gpt-4-turbo", label: "GPT-4 Turbo" },
  { value: "gpt-4.1", label: "GPT-4.1" },
  { value: "gpt-4.1-mini", label: "GPT-4.1 Mini" },
];

export function ExecutionAnalyticsForm() {
  const [results, setResults] = useState<ExecutionAnalyticsResponse | null>(
    null,
  );
  const { toast } = useToast();

  const generateAnalytics = usePostV2GenerateExecutionAnalytics({
    mutation: {
      onSuccess: (data) => {
        const result = data.data as ExecutionAnalyticsResponse;
        setResults(result);
        toast({
          title: "Analytics Generated",
          description: `Processed ${result.processed_executions} executions. ${result.successful_analytics} successful, ${result.failed_analytics} failed, ${result.skipped_executions} skipped.`,
          variant: "default",
        });
      },
      onError: (error: any) => {
        console.error("Analytics generation error:", error);
        toast({
          title: "Analytics Generation Failed",
          description:
            error?.message || error?.detail || "An unexpected error occurred",
          variant: "destructive",
        });
      },
    },
  });

  const [formData, setFormData] = useState<ExecutionAnalyticsRequest>({
    graph_id: "",
    model_name: "gpt-4o-mini",
    batch_size: 10, // Fixed internal value
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!formData.graph_id.trim()) {
      toast({
        title: "Validation Error",
        description: "Graph ID is required",
        variant: "destructive",
      });
      return;
    }

    setResults(null);

    // Prepare the request payload
    const payload: any = {
      graph_id: formData.graph_id.trim(),
      model_name: formData.model_name,
      batch_size: formData.batch_size,
    };

    if (formData.graph_version) {
      payload.graph_version = formData.graph_version;
    }

    if (formData.user_id?.trim()) {
      payload.user_id = formData.user_id.trim();
    }

    if (formData.created_after?.trim()) {
      payload.created_after = new Date(formData.created_after.trim());
    }

    // Use the generated API hook
    generateAnalytics.mutate({ data: payload });
  };

  const handleInputChange = (
    field: keyof ExecutionAnalyticsRequest,
    value: any,
  ) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  return (
    <div className="space-y-6">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <Label htmlFor="graph_id">
              Graph ID <span className="text-red-500">*</span>
            </Label>
            <Input
              id="graph_id"
              value={formData.graph_id}
              onChange={(e) => handleInputChange("graph_id", e.target.value)}
              placeholder="Enter graph/agent ID"
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="graph_version">Graph Version</Label>
            <Input
              id="graph_version"
              type="number"
              value={formData.graph_version || ""}
              onChange={(e) =>
                handleInputChange(
                  "graph_version",
                  e.target.value ? parseInt(e.target.value) : undefined,
                )
              }
              placeholder="Optional - leave empty for all versions"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="user_id">User ID</Label>
            <Input
              id="user_id"
              value={formData.user_id || ""}
              onChange={(e) => handleInputChange("user_id", e.target.value)}
              placeholder="Optional - leave empty for all users"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="created_after">Created After</Label>
            <Input
              id="created_after"
              type="datetime-local"
              value={formData.created_after || ""}
              onChange={(e) =>
                handleInputChange("created_after", e.target.value)
              }
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="model_name">AI Model</Label>
            <Select
              value={formData.model_name}
              onValueChange={(value) => handleInputChange("model_name", value)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select AI model" />
              </SelectTrigger>
              <SelectContent>
                {MODEL_OPTIONS.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="flex justify-end">
          <Button type="submit" disabled={generateAnalytics.isPending}>
            {generateAnalytics.isPending
              ? "Processing..."
              : "Generate Analytics"}
          </Button>
        </div>
      </form>

      {results && <AnalyticsResultsTable results={results} />}
    </div>
  );
}
