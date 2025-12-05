"use client";

import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/__legacy__/ui/input";
import { Label } from "@/components/__legacy__/ui/label";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { usePostV2GetExecutionAccuracyTrendsAndAlerts } from "@/app/api/__generated__/endpoints/admin/admin";
import type { AccuracyTrendsRequest } from "@/app/api/__generated__/models/accuracyTrendsRequest";
import type { AccuracyTrendsResponse } from "@/app/api/__generated__/models/accuracyTrendsResponse";

export function AccuracyTrendsChart() {
  const [formData, setFormData] = useState<AccuracyTrendsRequest>({
    graph_id: "",
    days_back: 30,
    drop_threshold: 10.0,
  });
  const [trendsData, setTrendsData] = useState<AccuracyTrendsResponse | null>(
    null,
  );
  const { toast } = useToast();

  // Use the generated API client for accuracy trends
  const accuracyTrendsMutation = usePostV2GetExecutionAccuracyTrendsAndAlerts({
    mutation: {
      onSuccess: (response) => {
        setTrendsData(response.data);

        if (response.data.alert) {
          toast({
            title: "🚨 Accuracy Alert Detected",
            description: `${response.data.alert.drop_percent.toFixed(1)}% accuracy drop detected for this agent`,
            variant: "destructive",
          });
        } else {
          toast({
            title: "Trends Loaded",
            description: "No accuracy alerts detected",
            variant: "default",
          });
        }
      },
      onError: (error: any) => {
        console.error("Failed to fetch trends:", error);
        toast({
          title: "Error",
          description: error?.message || "Failed to fetch accuracy trends",
          variant: "destructive",
        });
      },
    },
  });

  const fetchTrends = () => {
    if (!formData.graph_id.trim()) {
      toast({
        title: "Validation Error",
        description: "Graph ID is required",
        variant: "destructive",
      });
      return;
    }

    accuracyTrendsMutation.mutate({
      data: {
        graph_id: formData.graph_id.trim(),
        user_id: formData.user_id?.trim() || undefined,
        days_back: formData.days_back,
        drop_threshold: formData.drop_threshold,
      },
    });
  };

  return (
    <div className="space-y-6">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          fetchTrends();
        }}
        className="space-y-4"
      >
        <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
          <div className="space-y-2">
            <Label htmlFor="chart_graph_id">
              Graph ID <span className="text-red-500">*</span>
            </Label>
            <Input
              id="chart_graph_id"
              value={formData.graph_id}
              onChange={(e) =>
                setFormData((prev) => ({ ...prev, graph_id: e.target.value }))
              }
              placeholder="Enter graph/agent ID"
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="chart_user_id">User ID</Label>
            <Input
              id="chart_user_id"
              value={formData.user_id || ""}
              onChange={(e) =>
                setFormData((prev) => ({ ...prev, user_id: e.target.value }))
              }
              placeholder="Optional - all users"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="chart_days_back">
              Days Back
              <span className="ml-1 text-xs text-gray-500">
                (Min: 7, Max: 90)
              </span>
            </Label>
            <Input
              id="chart_days_back"
              type="number"
              value={formData.days_back}
              onChange={(e) =>
                setFormData((prev) => ({
                  ...prev,
                  days_back: parseInt(e.target.value) || 30,
                }))
              }
              min={7}
              max={90}
              placeholder="30"
            />
            <p className="text-xs text-gray-500">
              Historical period for trend analysis
            </p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="chart_threshold">
              Alert Threshold (%)
              <span className="ml-1 text-xs text-gray-500">
                (Recommended: 10-15%)
              </span>
            </Label>
            <Input
              id="chart_threshold"
              type="number"
              value={formData.drop_threshold}
              onChange={(e) =>
                setFormData((prev) => ({
                  ...prev,
                  drop_threshold: parseFloat(e.target.value) || 10.0,
                }))
              }
              min={1}
              max={50}
              step={0.1}
              placeholder="10.0"
            />
            <p className="text-xs text-gray-500">
              Alert when 3-day avg drops this % below 7-day avg
            </p>
          </div>
        </div>

        <div className="flex justify-end">
          <Button
            type="submit"
            variant="primary"
            size="medium"
            disabled={accuracyTrendsMutation.isPending}
          >
            {accuracyTrendsMutation.isPending ? "Loading..." : "Load Trends"}
          </Button>
        </div>
      </form>

      {trendsData && (
        <div className="space-y-4">
          {/* Alert Section */}
          {trendsData.alert && (
            <div className="rounded-lg border-l-4 border-red-500 bg-red-50 p-4">
              <div className="flex items-start">
                <span className="text-2xl">🚨</span>
                <div className="ml-3 space-y-2">
                  <h3 className="text-lg font-semibold text-red-800">
                    Accuracy Alert Detected
                  </h3>
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
                      • Alert threshold:{" "}
                      <strong>{formData.drop_threshold}%</strong>
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

          {/* Chart Placeholder */}
          <div className="rounded-lg border bg-white p-6">
            <h3 className="mb-4 text-lg font-semibold">
              Execution Accuracy Trends
            </h3>
            <div className="flex h-64 items-center justify-center rounded bg-gray-50">
              <div className="text-center text-gray-500">
                <p className="text-lg">📊 Chart View</p>
                <p className="mt-2 text-sm">
                  Latest data point shown above.
                  <br />
                  Full historical chart coming soon.
                </p>
                <div className="mt-4 text-xs text-amber-600">
                  💡 For testing: Use graph ID{" "}
                  <code className="rounded bg-amber-100 px-1">
                    test-accuracy-agent-001
                  </code>
                </div>
              </div>
            </div>
          </div>

          {/* Data Details */}
          <div className="rounded-lg border bg-gray-50 p-4">
            <h4 className="mb-2 font-semibold text-gray-700">
              Latest Data Point
            </h4>
            <div className="text-sm text-gray-600">
              <p>
                <strong>Date:</strong>{" "}
                {new Date(trendsData.latest_data.date).toLocaleDateString()}
              </p>
              <p>
                <strong>Graph ID:</strong> {formData.graph_id}
              </p>
              {formData.user_id && (
                <p>
                  <strong>User ID:</strong> {formData.user_id}
                </p>
              )}
              <p>
                <strong>Alert Threshold:</strong> {formData.drop_threshold}%
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
