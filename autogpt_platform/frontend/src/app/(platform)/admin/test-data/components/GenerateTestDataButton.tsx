"use client";

import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Select, SelectOption } from "@/components/atoms/Select/Select";
import { Text } from "@/components/atoms/Text/Text";
import { useToast } from "@/components/molecules/Toast/use-toast";
// Generated types and hooks from OpenAPI spec
// Run `npm run generate:api` to regenerate after backend changes
import { usePostAdminGenerateTestData } from "@/app/api/__generated__/endpoints/admin/admin";
import type { GenerateTestDataResponse } from "@/app/api/__generated__/models/generateTestDataResponse";
import type { TestDataScriptType } from "@/app/api/__generated__/models/testDataScriptType";

const scriptTypeOptions: SelectOption[] = [
  {
    value: "e2e",
    label:
      "E2E Test Data - 15 users with graphs, agents, and store submissions",
  },
  {
    value: "full",
    label: "Full Test Data - 100+ users with comprehensive data (takes longer)",
  },
];

export function GenerateTestDataButton() {
  const { toast } = useToast();
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [scriptType, setScriptType] = useState<TestDataScriptType>("e2e");
  const [result, setResult] = useState<GenerateTestDataResponse | null>(null);

  const generateMutation = usePostAdminGenerateTestData({
    mutation: {
      onSuccess: (response) => {
        const data = response.data;
        setResult(data);
        if (data.success) {
          toast({
            title: "Success",
            description: data.message,
          });
        } else {
          toast({
            title: "Error",
            description: data.message,
            variant: "destructive",
          });
        }
      },
      onError: (error) => {
        console.error("Error generating test data:", error);
        const errorMessage =
          error instanceof Error ? error.message : "Unknown error occurred";
        setResult({
          success: false,
          message: `Failed to generate test data: ${errorMessage}`,
        });
        toast({
          title: "Error",
          description: "Failed to generate test data. Please try again.",
          variant: "destructive",
        });
      },
    },
  });

  const handleGenerate = () => {
    setResult(null);
    generateMutation.mutate({
      data: {
        script_type: scriptType,
      },
    });
  };

  const handleDialogClose = () => {
    setIsDialogOpen(false);
  };

  return (
    <>
      <Button
        size="large"
        variant="primary"
        onClick={() => {
          setIsDialogOpen(true);
          setResult(null);
        }}
      >
        Generate Test Data
      </Button>

      <Dialog
        title="Generate Test Data"
        controlled={{
          isOpen: isDialogOpen,
          set: (open) => {
            if (!open) handleDialogClose();
          },
        }}
        styling={{ maxWidth: "32rem" }}
      >
        <Dialog.Content>
          <Text variant="body" className="pb-4 text-neutral-600">
            This will populate the database with sample test data including
            users, agents, graphs, store listings, and more.
          </Text>

          <div className="grid gap-4 py-4">
            <Select
              label="Script Type"
              id="scriptType"
              value={scriptType}
              onValueChange={(value) =>
                setScriptType(value as TestDataScriptType)
              }
              disabled={generateMutation.isPending}
              options={scriptTypeOptions}
            />

            <div className="rounded-md bg-yellow-50 p-3 text-yellow-800">
              <Text variant="small" as="span">
                <Text variant="small-medium" as="span">
                  Warning:
                </Text>{" "}
                This will add significant data to your database. This endpoint
                is disabled in production environments.
              </Text>
            </div>

            {result && (
              <div
                className={`rounded-md p-3 ${
                  result.success
                    ? "bg-green-50 text-green-800"
                    : "bg-red-50 text-red-800"
                }`}
              >
                <Text variant="small-medium">{result.message}</Text>
                {result.details && (
                  <ul className="mt-2 list-inside list-disc">
                    {Object.entries(result.details).map(([key, value]) => (
                      <li key={key}>
                        <Text variant="small" as="span">
                          {key.replace(/_/g, " ")}: {String(value)}
                        </Text>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </div>

          <Dialog.Footer>
            <Button
              variant="outline"
              onClick={handleDialogClose}
              disabled={generateMutation.isPending}
            >
              Cancel
            </Button>
            <Button
              variant="primary"
              onClick={handleGenerate}
              disabled={generateMutation.isPending}
              loading={generateMutation.isPending}
            >
              {generateMutation.isPending
                ? "Generating..."
                : "Generate Test Data"}
            </Button>
          </Dialog.Footer>
        </Dialog.Content>
      </Dialog>
    </>
  );
}
