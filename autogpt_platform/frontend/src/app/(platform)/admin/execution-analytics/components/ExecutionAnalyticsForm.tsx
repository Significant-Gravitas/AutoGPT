"use client";

// TODO(tokens): no role for these colours yet, so they stay raw:
// bg-gray-100 #F3F4F6, bg-red-100 #FDECEC, text-blue-700 #1D4ED8,
// text-gray-500 #6B7280, text-gray-600 #4B5563, text-gray-700 #374151,
// text-orange-600 #E77B00, text-purple-600 #6C2EDF, text-red-600 #D93636,
// text-red-800 #832525.
// Same value as a role with another meaning, so not swapped:
// text-red-600 = error-hover (a hover fill),
// text-purple-600 = brand-hover (a hover fill).
// TODO(tokens): chart and canvas colours, which have no tokens yet: #3B82F6,
// #10B981, #F59E0B, #8B5CF6.

import { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Button } from "@/components/atoms/Button/Button";
import { BaseInput } from "@/components/atoms/Input/BaseInput";
import { Text } from "@/components/atoms/Text/Text";
import { Label } from "@/components/atoms/Label/Label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/atoms/Select/components/SelectPrimitives";
import { Textarea } from "@/components/atoms/Textarea/Textarea";
import { Checkbox } from "@/components/atoms/Checkbox/Checkbox";
import { Collapsible } from "@/components/molecules/Collapsible/Collapsible";
import { useToast } from "@/components/molecules/Toast/use-toast";
import {
  usePostV2GenerateExecutionAnalytics,
  useGetV2GetExecutionAnalyticsConfiguration,
  useGetV2GetExecutionAccuracyTrendsAndAlerts,
} from "@/app/api/__generated__/endpoints/admin/admin";
import type { ExecutionAnalyticsRequest } from "@/app/api/__generated__/models/executionAnalyticsRequest";
import type { ExecutionAnalyticsResponse } from "@/app/api/__generated__/models/executionAnalyticsResponse";
import type { AccuracyTrendsResponse } from "@/app/api/__generated__/models/accuracyTrendsResponse";
import type { AccuracyLatestData } from "@/app/api/__generated__/models/accuracyLatestData";
import { toAccuracyChartData } from "./accuracyChartData";

// Use the generated type with minimal adjustment for form handling
interface FormData extends Omit<ExecutionAnalyticsRequest, "created_after"> {
  created_after?: string; // Keep as string for datetime-local input
  // All other fields use the generated types as-is
}
import { AnalyticsResultsTable } from "./AnalyticsResultsTable";
import { okData } from "@/app/api/helpers";

export function ExecutionAnalyticsForm() {
  const [results, setResults] = useState<ExecutionAnalyticsResponse | null>(
    null,
  );
  const [trendsData, setTrendsData] = useState<AccuracyTrendsResponse | null>(
    null,
  );
  const { toast } = useToast();

  // State for accuracy trends query parameters
  const [accuracyParams, setAccuracyParams] = useState<{
    graph_id: string;
    user_id?: string;
    days_back: number;
    drop_threshold: number;
    include_historical?: boolean;
  } | null>(null);

  // Use the generated API client for accuracy trends (GET)
  const { data: accuracyApiResponse, error: accuracyError } =
    useGetV2GetExecutionAccuracyTrendsAndAlerts(
      accuracyParams || {
        graph_id: "",
        days_back: 30,
        drop_threshold: 10.0,
        include_historical: false,
      },
      {
        query: {
          enabled: !!accuracyParams?.graph_id,
        },
      },
    );

  // Update local state when data changes and handle success/error
  useEffect(() => {
    if (accuracyError) {
      console.error("Failed to fetch trends:", accuracyError);
      toast({
        title: "Trends Error",
        description:
          (accuracyError as any)?.message || "Failed to fetch accuracy trends",
        variant: "destructive",
      });
      return;
    }

    const data = accuracyApiResponse?.data;
    if (data && "latest_data" in data) {
      setTrendsData(data);

      // Check for alerts
      if (data.alert) {
        toast({
          title: "🚨 Accuracy Alert Detected",
          description: `${data.alert.drop_percent.toFixed(1)}% accuracy drop detected for this agent`,
          variant: "destructive",
        });
      }
    }
  }, [accuracyApiResponse, accuracyError, toast]);

  // Chart component for accuracy trends
  function AccuracyChart({ data }: { data: AccuracyLatestData[] }) {
    const chartData = toAccuracyChartData(data);

    return (
      <ResponsiveContainer width="100%" height={400}>
        <LineChart
          data={chartData}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis domain={[0, 100]} />
          <Tooltip
            formatter={(value) => [`${Number(value).toFixed(2)}%`, ""]}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="Daily Score"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={{ r: 3 }}
          />
          <Line
            type="monotone"
            dataKey="3-Day Avg"
            stroke="#10b981"
            strokeWidth={2}
            dot={{ r: 3 }}
          />
          <Line
            type="monotone"
            dataKey="7-Day Avg"
            stroke="#f59e0b"
            strokeWidth={2}
            dot={{ r: 3 }}
          />
          <Line
            type="monotone"
            dataKey="14-Day Avg"
            stroke="#8b5cf6"
            strokeWidth={2}
            dot={{ r: 3 }}
          />
        </LineChart>
      </ResponsiveContainer>
    );
  }

  // Function to fetch accuracy trends using generated API client
  const fetchAccuracyTrends = (graphId: string, userId?: string) => {
    if (!graphId.trim()) return;

    setAccuracyParams({
      graph_id: graphId.trim(),
      user_id: userId?.trim() || undefined,
      days_back: 30,
      drop_threshold: 10.0,
      include_historical: showAccuracyChart, // Include historical data when chart is enabled
    });
  };

  // Fetch configuration from API
  const {
    data: config,
    isLoading: configLoading,
    error: configError,
  } = useGetV2GetExecutionAnalyticsConfiguration({ query: { select: okData } });

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

        const errorMessage =
          error?.message || error?.detail || "An unexpected error occurred";
        const isOpenAIError = errorMessage.includes(
          "OpenAI API key not configured",
        );

        toast({
          title: isOpenAIError
            ? "Analytics Generation Skipped"
            : "Analytics Generation Failed",
          description: isOpenAIError
            ? "Analytics generation requires OpenAI configuration, but accuracy trends are still available above."
            : errorMessage,
          variant: isOpenAIError ? "default" : "destructive",
        });
      },
    },
  });

  const [formData, setFormData] = useState<FormData>({
    graph_id: "",
    model_name: "", // Will be set from config
    batch_size: 10, // Fixed internal value
    skip_existing: true, // Default to skip existing
    system_prompt: "", // Will use config default when empty
    user_prompt: "", // Will use config default when empty
  });

  // State for accuracy trends chart toggle
  const [showAccuracyChart, setShowAccuracyChart] = useState(true);

  // Update form defaults when config loads
  useEffect(() => {
    if (config && !formData.model_name) {
      setFormData((prev) => ({
        ...prev,
        model_name: config.recommended_model,
      }));
    }
  }, [config, formData.model_name]);

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

    // Fetch accuracy trends if chart is enabled
    if (showAccuracyChart) {
      fetchAccuracyTrends(formData.graph_id, formData.user_id || undefined);
    }

    // Prepare the request payload
    const payload: ExecutionAnalyticsRequest = {
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

  // Show loading state while config loads
  if (configLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <div className="text-gray-500">Loading configuration...</div>
      </div>
    );
  }

  // Show error state if config fails to load
  if (configError || !config) {
    return (
      <div className="flex items-center justify-center py-8">
        <div className="text-error">Failed to load configuration</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <Label htmlFor="graph_id">
              Graph ID <span className="text-error">*</span>
            </Label>
            <BaseInput
              id="graph_id"
              value={formData.graph_id}
              onChange={(e) => handleInputChange("graph_id", e.target.value)}
              placeholder="Enter graph/agent ID"
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="graph_version">Graph Version</Label>
            <BaseInput
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
            <BaseInput
              id="user_id"
              value={formData.user_id || ""}
              onChange={(e) => handleInputChange("user_id", e.target.value)}
              placeholder="Optional - leave empty for all users"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="created_after">Created After</Label>
            <BaseInput
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
                {config.available_models.map((model) => (
                  <SelectItem key={model.value} value={model.value}>
                    {model.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* Advanced Options Section - Collapsible */}
        <div className="border-t pt-6">
          <Collapsible
            trigger={
              <Text variant="lg-semibold" as="h3" className="text-gray-700">
                Advanced Options
              </Text>
            }
            defaultOpen={false}
            className="space-y-4"
          >
            <div className="space-y-4 pt-4">
              {/* Skip Existing Checkbox */}
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="skip_existing"
                  checked={formData.skip_existing}
                  onCheckedChange={(checked) =>
                    handleInputChange("skip_existing", checked)
                  }
                />
                <Label htmlFor="skip_existing">
                  <Text variant="body-medium-tight" as="span" tone="inherit">
                    Skip executions that already have activity status and
                    correctness score
                  </Text>
                </Label>
              </div>

              {/* Show Accuracy Chart Checkbox */}
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="show_accuracy_chart"
                  checked={showAccuracyChart}
                  onCheckedChange={(checked) => setShowAccuracyChart(!!checked)}
                />
                <Label htmlFor="show_accuracy_chart">
                  <Text variant="body-medium-tight" as="span" tone="inherit">
                    Show accuracy trends chart and historical data visualization
                  </Text>
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
                  placeholder={config.default_system_prompt}
                  rows={6}
                  className="resize-y"
                />
                <Text variant="body-tight" className="text-gray-600">
                  Customize how the AI evaluates execution success and failure.
                  Leave empty to use the default prompt shown above.
                </Text>
              </div>

              {/* Custom User Prompt */}
              <div className="space-y-2">
                <Label htmlFor="user_prompt">
                  Custom User Prompt Template (Optional)
                </Label>
                <Textarea
                  id="user_prompt"
                  value={formData.user_prompt || ""}
                  onChange={(e) =>
                    handleInputChange("user_prompt", e.target.value)
                  }
                  placeholder={config.default_user_prompt}
                  rows={8}
                  className="resize-y"
                />
                <Text variant="body-tight" className="text-gray-600">
                  Customize the analysis instructions. Use{" "}
                  <code className="rounded bg-gray-100 px-1">
                    {"{{GRAPH_NAME}}"}
                  </code>{" "}
                  and{" "}
                  <code className="rounded bg-gray-100 px-1">
                    {"{{EXECUTION_DATA}}"}
                  </code>{" "}
                  as placeholders. Leave empty to use the default template shown
                  above.
                </Text>
              </div>

              {/* Quick Actions */}
              <div className="flex flex-wrap gap-2 border-t pt-4">
                <Button
                  type="button"
                  variant="secondary"
                  size="small"
                  onClick={() => {
                    handleInputChange(
                      "system_prompt",
                      config.default_system_prompt,
                    );
                  }}
                >
                  Reset System Prompt
                </Button>
                <Button
                  type="button"
                  variant="secondary"
                  size="small"
                  onClick={() => {
                    handleInputChange(
                      "user_prompt",
                      config.default_user_prompt,
                    );
                  }}
                >
                  Reset User Prompt
                </Button>
                <Button
                  type="button"
                  variant="secondary"
                  size="small"
                  onClick={() => {
                    handleInputChange("system_prompt", "");
                    handleInputChange("user_prompt", "");
                  }}
                >
                  Clear All Prompts
                </Button>
              </div>
            </div>
          </Collapsible>
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

      {/* Accuracy Trends Display */}
      {trendsData && (
        <div className="space-y-4">
          <div className="flex items-start justify-between">
            <Text variant="lg-semibold" as="h3" tone="inherit">
              Execution Accuracy Trends
            </Text>
            <Text
              variant="small-tight"
              as="div"
              className="rounded-md bg-info-surface px-3 py-2 text-blue-700"
            >
              <p className="font-medium">
                Chart Filters (matches monitoring system):
              </p>
              <ul className="mt-1 list-inside list-disc space-y-1">
                <li>Only days with ≥1 execution with correctness score</li>
                <li>Last 30 days</li>
                <li>Averages calculated from scored executions only</li>
              </ul>
            </Text>
          </div>

          {/* Alert Section */}
          {trendsData.alert && (
            <div className="rounded-lg border-l-4 border-error bg-error-surface p-4">
              <div className="flex items-start">
                <Text variant="2xl" as="span" tone="inherit">
                  🚨
                </Text>
                <div className="ml-3 space-y-2">
                  <Text variant="lg-semibold" as="h4" className="text-red-800">
                    Accuracy Alert Detected
                  </Text>
                  <p className="text-error-foreground">
                    <strong>
                      {trendsData.alert.drop_percent.toFixed(1)}% accuracy drop
                    </strong>{" "}
                    detected for agent{" "}
                    <Text
                      variant="body-mono-tight"
                      as="code"
                      tone="inherit"
                      className="rounded bg-red-100 px-1"
                    >
                      {formData.graph_id}
                    </Text>
                  </p>
                  <Text
                    variant="body-tight"
                    as="div"
                    className="space-y-1 text-red-600"
                  >
                    <p>
                      • 3-day average:{" "}
                      <strong>
                        {trendsData.alert.three_day_avg.toFixed(2)}%
                      </strong>
                    </p>
                    <p>
                      • 7-day average:{" "}
                      <strong>
                        {trendsData.alert.seven_day_avg.toFixed(2)}%
                      </strong>
                    </p>
                    <p>
                      • Detected at:{" "}
                      <strong>
                        {new Date(
                          trendsData.alert.detected_at,
                        ).toLocaleString()}
                      </strong>
                    </p>
                  </Text>
                </div>
              </div>
            </div>
          )}

          {/* Latest Data Summary */}
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <div className="rounded-lg border bg-surface p-4 text-center">
              <Text
                variant="2xl-bold"
                as="div"
                className="text-info-foreground"
              >
                {trendsData.latest_data.daily_score?.toFixed(2) || "N/A"}
              </Text>
              <Text variant="body-tight" as="div" className="text-gray-600">
                Daily Score
              </Text>
            </div>
            <div className="rounded-lg border bg-surface p-4 text-center">
              <Text variant="2xl-bold" as="div" tone="success">
                {trendsData.latest_data.three_day_avg?.toFixed(2) || "N/A"}
              </Text>
              <Text variant="body-tight" as="div" className="text-gray-600">
                3-Day Avg
              </Text>
            </div>
            <div className="rounded-lg border bg-surface p-4 text-center">
              <Text variant="2xl-bold" as="div" className="text-orange-600">
                {trendsData.latest_data.seven_day_avg?.toFixed(2) || "N/A"}
              </Text>
              <Text variant="body-tight" as="div" className="text-gray-600">
                7-Day Avg
              </Text>
            </div>
            <div className="rounded-lg border bg-surface p-4 text-center">
              <Text variant="2xl-bold" as="div" className="text-purple-600">
                {trendsData.latest_data.fourteen_day_avg?.toFixed(2) || "N/A"}
              </Text>
              <Text variant="body-tight" as="div" className="text-gray-600">
                14-Day Avg
              </Text>
            </div>
          </div>

          {/* Chart Section - only show when toggle is enabled and historical data exists */}
          {showAccuracyChart && trendsData?.historical_data && (
            <div className="mt-6">
              <Text
                variant="lg-semibold"
                as="h4"
                tone="inherit"
                className="mb-4"
              >
                Execution Accuracy Trends Chart
              </Text>
              <div className="rounded-lg border bg-surface p-6">
                <AccuracyChart data={trendsData.historical_data} />
              </div>
            </div>
          )}
        </div>
      )}

      {results && <AnalyticsResultsTable results={results} />}
    </div>
  );
}
