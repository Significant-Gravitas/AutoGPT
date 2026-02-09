"use client";

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
    const chartData = data.map((item) => ({
      date: new Date(item.date).toLocaleDateString(),
      "Daily Score": item.daily_score,
      "3-Day Avg": item.three_day_avg,
      "7-Day Avg": item.seven_day_avg,
      "14-Day Avg": item.fourteen_day_avg,
    }));

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
        <div className="text-red-500">Failed to load configuration</div>
      </div>
    );
  }

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
              <h3 className="text-lg font-semibold text-gray-700">
                Advanced Options
              </h3>
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
                <Label htmlFor="skip_existing" className="text-sm">
                  Skip executions that already have activity status and
                  correctness score
                </Label>
              </div>

              {/* Show Accuracy Chart Checkbox */}
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="show_accuracy_chart"
                  checked={showAccuracyChart}
                  onCheckedChange={(checked) => setShowAccuracyChart(!!checked)}
                />
                <Label htmlFor="show_accuracy_chart" className="text-sm">
                  Show accuracy trends chart and historical data visualization
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
                <p className="text-sm text-gray-600">
                  Customize how the AI evaluates execution success and failure.
                  Leave empty to use the default prompt shown above.
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
                  onChange={(e) =>
                    handleInputChange("user_prompt", e.target.value)
                  }
                  placeholder={config.default_user_prompt}
                  rows={8}
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
                  as placeholders. Leave empty to use the default template shown
                  above.
                </p>
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
            <h3 className="text-lg font-semibold">Execution Accuracy Trends</h3>
            <div className="rounded-md bg-blue-50 px-3 py-2 text-xs text-blue-700">
              <p className="font-medium">
                Chart Filters (matches monitoring system):
              </p>
              <ul className="mt-1 list-inside list-disc space-y-1">
                <li>Only days with ≥1 execution with correctness score</li>
                <li>Last 30 days</li>
                <li>Averages calculated from scored executions only</li>
              </ul>
            </div>
          </div>

          {/* Alert Section */}
          {trendsData.alert && (
            <div className="rounded-lg border-l-4 border-red-500 bg-red-50 p-4">
              <div className="flex items-start">
                <span className="text-2xl">🚨</span>
                <div className="ml-3 space-y-2">
                  <h4 className="text-lg font-semibold text-red-800">
                    Accuracy Alert Detected
                  </h4>
                  <p className="text-red-700">
                    <strong>
                      {trendsData.alert.drop_percent.toFixed(1)}% accuracy drop
                    </strong>{" "}
                    detected for agent{" "}
                    <code className="rounded bg-red-100 px-1 text-sm">
                      {formData.graph_id}
                    </code>
                  </p>
                  <div className="space-y-1 text-sm text-red-600">
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
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Latest Data Summary */}
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <div className="rounded-lg border bg-white p-4 text-center">
              <div className="text-2xl font-bold text-blue-600">
                {trendsData.latest_data.daily_score?.toFixed(2) || "N/A"}
              </div>
              <div className="text-sm text-gray-600">Daily Score</div>
            </div>
            <div className="rounded-lg border bg-white p-4 text-center">
              <div className="text-2xl font-bold text-green-600">
                {trendsData.latest_data.three_day_avg?.toFixed(2) || "N/A"}
              </div>
              <div className="text-sm text-gray-600">3-Day Avg</div>
            </div>
            <div className="rounded-lg border bg-white p-4 text-center">
              <div className="text-2xl font-bold text-orange-600">
                {trendsData.latest_data.seven_day_avg?.toFixed(2) || "N/A"}
              </div>
              <div className="text-sm text-gray-600">7-Day Avg</div>
            </div>
            <div className="rounded-lg border bg-white p-4 text-center">
              <div className="text-2xl font-bold text-purple-600">
                {trendsData.latest_data.fourteen_day_avg?.toFixed(2) || "N/A"}
              </div>
              <div className="text-sm text-gray-600">14-Day Avg</div>
            </div>
          </div>

          {/* Chart Section - only show when toggle is enabled and historical data exists */}
          {showAccuracyChart && trendsData?.historical_data && (
            <div className="mt-6">
              <h4 className="mb-4 text-lg font-semibold">
                Execution Accuracy Trends Chart
              </h4>
              <div className="rounded-lg border bg-white p-6">
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
