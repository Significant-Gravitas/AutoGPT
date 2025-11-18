"use client";

import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/__legacy__/ui/input";
import { Label } from "@/components/__legacy__/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/__legacy__/ui/select";
import { Textarea } from "@/components/__legacy__/ui/textarea";
import { Checkbox } from "@/components/__legacy__/ui/checkbox";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { usePostV2GenerateExecutionAnalytics } from "@/app/api/__generated__/endpoints/admin/admin";
import type { ExecutionAnalyticsRequest } from "@/app/api/__generated__/models/executionAnalyticsRequest";
import type { ExecutionAnalyticsResponse } from "@/app/api/__generated__/models/executionAnalyticsResponse";

// Local interface for form state to simplify handling
interface FormData {
  graph_id: string;
  graph_version?: number;
  user_id?: string;
  created_after?: string;
  model_name: string;
  batch_size: number;
  system_prompt?: string;
  user_prompt?: string;
  skip_existing: boolean;
}
import { AnalyticsResultsTable } from "./AnalyticsResultsTable";

const MODEL_OPTIONS = [
  { value: "gpt-4o-mini", label: "GPT-4o Mini (Recommended)" },
  { value: "gpt-4o", label: "GPT-4o" },
  { value: "gpt-4-turbo", label: "GPT-4 Turbo" },
  { value: "gpt-4.1-2025-04-14", label: "GPT-4.1" },
  { value: "gpt-4.1-mini-2025-04-14", label: "GPT-4.1 Mini" },
  { value: "claude-sonnet-4-5-20250929", label: "Claude 4.5 Sonnet" },
  { value: "claude-haiku-4-5-20251001", label: "Claude 4.5 Haiku" },
  { value: "claude-opus-4-1-20250805", label: "Claude 4.1 Opus" },
  { value: "gpt-5-2025-08-07", label: "GPT-5" },
  { value: "gpt-5-mini-2025-08-07", label: "GPT-5 Mini" },
];

export function ExecutionAnalyticsForm() {
  const [results, setResults] = useState<ExecutionAnalyticsResponse | null>(
    null,
  );
  const { toast } = useToast();

  const generateAnalytics = usePostV2GenerateExecutionAnalytics({
    mutation: {
      onSuccess: (res) => {
        if (res.status !== 200) {
          throw new Error("Something went wrong!");
        }
        const result = res.data;
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

  const [formData, setFormData] = useState<FormData>({
    graph_id: "",
    model_name: "gpt-4o-mini",
    batch_size: 10, // Fixed internal value
    skip_existing: true, // Default to skip existing
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
      skip_existing: formData.skip_existing,
    };

    if (formData.graph_version) {
      payload.graph_version = formData.graph_version;
    }

    if (formData.user_id?.trim()) {
      payload.user_id = formData.user_id.trim();
    }

    if (
      formData.created_after &&
      typeof formData.created_after === "string" &&
      formData.created_after.trim()
    ) {
      payload.created_after = new Date(formData.created_after.trim());
    }

    if (formData.system_prompt?.trim()) {
      payload.system_prompt = formData.system_prompt.trim();
    }

    if (formData.user_prompt?.trim()) {
      payload.user_prompt = formData.user_prompt.trim();
    }

    generateAnalytics.mutate({ data: payload });
  };

  const handleInputChange = (field: keyof FormData, value: any) => {
    setFormData((prev: FormData) => ({ ...prev, [field]: value }));
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

        {/* Advanced Options Section */}
        <div className="space-y-4 border-t pt-6">
          <h3 className="text-lg font-semibold text-gray-700">
            Advanced Options
          </h3>

          {/* Skip Existing Checkbox */}
          <div className="flex items-center space-x-2">
            <Checkbox
              id="skip_existing"
              checked={formData.skip_existing}
              onCheckedChange={(checked) =>
                handleInputChange("skip_existing", checked)
              }
            />
            <Label htmlFor="skip_existing" className="text-sm">
              Skip executions that already have activity status and correctness
              score
            </Label>
          </div>

          {/* Custom System Prompt */}
          <div className="space-y-2">
            <Label htmlFor="system_prompt">
              Custom System Prompt (Optional)
            </Label>
            <Textarea
              id="system_prompt"
              value={formData.system_prompt || ""}
              onChange={(e) =>
                handleInputChange("system_prompt", e.target.value)
              }
              placeholder="Leave empty to use default system prompt. This defines the AI evaluation criteria..."
              rows={4}
              className="resize-y"
            />
            <p className="text-sm text-gray-600">
              Customize how the AI evaluates execution success and failure.
              Leave empty to use the built-in prompt.
            </p>
          </div>

          {/* Custom User Prompt */}
          <div className="space-y-2">
            <Label htmlFor="user_prompt">
              Custom User Prompt Template (Optional)
            </Label>
            <Textarea
              id="user_prompt"
              value={formData.user_prompt || ""}
              onChange={(e) => handleInputChange("user_prompt", e.target.value)}
              placeholder="Leave empty to use default template. Use {{GRAPH_NAME}} and {{EXECUTION_DATA}} as placeholders..."
              rows={6}
              className="resize-y"
            />
            <p className="text-sm text-gray-600">
              Customize the analysis instructions. Use{" "}
              <code className="rounded bg-gray-100 px-1">
                {"{{GRAPH_NAME}}"}
              </code>{" "}
              and{" "}
              <code className="rounded bg-gray-100 px-1">
                {"{{EXECUTION_DATA}}"}
              </code>{" "}
              as placeholders. Leave empty to use the built-in template.
            </p>
          </div>
        </div>

        <div className="flex justify-end">
          <Button
            variant="primary"
            size="large"
            type="submit"
            disabled={generateAnalytics.isPending}
          >
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
