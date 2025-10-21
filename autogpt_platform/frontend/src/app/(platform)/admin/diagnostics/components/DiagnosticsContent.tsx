"use client";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ArrowClockwise } from "@phosphor-icons/react";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { useDiagnosticsContent } from "./useDiagnosticsContent";

export function DiagnosticsContent() {
  const { executionData, agentData, isLoading, isError, error, refresh } =
    useDiagnosticsContent();

  if (isLoading && !executionData && !agentData) {
    return (
      <div className="flex h-64 items-center justify-center">
        <div className="text-center">
          <ArrowClockwise className="mx-auto h-8 w-8 animate-spin text-gray-400" />
          <p className="mt-2 text-gray-500">Loading diagnostics...</p>
        </div>
      </div>
    );
  }

  if (isError) {
    return (
      <ErrorCard
        httpError={error as any}
        onRetry={refresh}
        context="diagnostics"
      />
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">System Diagnostics</h1>
          <p className="text-gray-500">
            Monitor execution and agent system health
          </p>
        </div>
        <Button
          onClick={refresh}
          disabled={isLoading}
          variant="outline"
          size="sm"
        >
          <ArrowClockwise
            className={`mr-2 h-4 w-4 ${isLoading ? "animate-spin" : ""}`}
          />
          Refresh
        </Button>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Execution Diagnostics</CardTitle>
            <CardDescription>
              Monitor execution queue and processing status
            </CardDescription>
          </CardHeader>
          <CardContent>
            {executionData ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <p className="text-sm font-medium text-gray-500">
                      Running Executions
                    </p>
                    <p className="text-3xl font-bold">
                      {executionData.running_executions}
                    </p>
                  </div>
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-green-100">
                    <div className="h-6 w-6 rounded-full bg-green-500"></div>
                  </div>
                </div>

                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <p className="text-sm font-medium text-gray-500">
                      Queued in Database
                    </p>
                    <p className="text-3xl font-bold">
                      {executionData.queued_executions_db}
                    </p>
                  </div>
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-blue-100">
                    <div className="h-6 w-6 rounded-full bg-blue-500"></div>
                  </div>
                </div>

                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <p className="text-sm font-medium text-gray-500">
                      Queued in RabbitMQ
                    </p>
                    <p className="text-3xl font-bold">
                      {executionData.queued_executions_rabbitmq === -1 ? (
                        <span className="text-xl text-red-500">Error</span>
                      ) : (
                        executionData.queued_executions_rabbitmq
                      )}
                    </p>
                  </div>
                  <div
                    className={`flex h-12 w-12 items-center justify-center rounded-full ${
                      executionData.queued_executions_rabbitmq === -1
                        ? "bg-red-100"
                        : "bg-yellow-100"
                    }`}
                  >
                    <div
                      className={`h-6 w-6 rounded-full ${
                        executionData.queued_executions_rabbitmq === -1
                          ? "bg-red-500"
                          : "bg-yellow-500"
                      }`}
                    ></div>
                  </div>
                </div>

                <div className="text-xs text-gray-400">
                  Last updated:{" "}
                  {new Date(executionData.timestamp).toLocaleString()}
                </div>
              </div>
            ) : (
              <p className="text-gray-500">No data available</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Agent Diagnostics</CardTitle>
            <CardDescription>
              Monitor agent system status and activity
            </CardDescription>
          </CardHeader>
          <CardContent>
            {agentData ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <p className="text-sm font-medium text-gray-500">
                      Total Agents
                    </p>
                    <p className="text-3xl font-bold">
                      {agentData.total_agents}
                    </p>
                  </div>
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-purple-100">
                    <div className="h-6 w-6 rounded-full bg-purple-500"></div>
                  </div>
                </div>

                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <p className="text-sm font-medium text-gray-500">
                      Active Agents
                    </p>
                    <p className="text-3xl font-bold">
                      {agentData.active_agents}
                    </p>
                  </div>
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-indigo-100">
                    <div className="h-6 w-6 rounded-full bg-indigo-500"></div>
                  </div>
                </div>

                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <p className="text-sm font-medium text-gray-500">
                      Agents with Active Executions
                    </p>
                    <p className="text-3xl font-bold">
                      {agentData.agents_with_active_executions}
                    </p>
                  </div>
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-teal-100">
                    <div className="h-6 w-6 rounded-full bg-teal-500"></div>
                  </div>
                </div>

                <div className="text-xs text-gray-400">
                  Last updated: {new Date(agentData.timestamp).toLocaleString()}
                </div>
              </div>
            ) : (
              <p className="text-gray-500">No data available</p>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Diagnostic Information</CardTitle>
          <CardDescription>
            Understanding the metrics for on-call diagnostics
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3 text-sm">
            <div>
              <p className="font-semibold">Running Executions:</p>
              <p className="text-gray-600">
                Number of executions currently being processed by workers.
                Higher numbers indicate active system load.
              </p>
            </div>
            <div>
              <p className="font-semibold">Queued in Database:</p>
              <p className="text-gray-600">
                Executions marked as QUEUED in the database. These are waiting
                to be picked up by workers.
              </p>
            </div>
            <div>
              <p className="font-semibold">Queued in RabbitMQ:</p>
              <p className="text-gray-600">
                Number of messages in the RabbitMQ execution queue. This
                represents the backlog of work waiting to be processed. If this
                shows "Error", there may be an issue connecting to RabbitMQ.
              </p>
            </div>
            <div>
              <p className="font-semibold">Agents with Active Executions:</p>
              <p className="text-gray-600">
                Number of unique agents that currently have running or queued
                executions. Helps identify which agents are being actively used.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
