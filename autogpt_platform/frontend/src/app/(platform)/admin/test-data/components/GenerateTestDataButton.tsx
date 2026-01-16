"use client";

import { useState } from "react";
import { Button } from "@/components/__legacy__/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/__legacy__/ui/dialog";
import { Label } from "@/components/__legacy__/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/__legacy__/ui/select";
import { useToast } from "@/components/molecules/Toast/use-toast";
import BackendAPI from "@/lib/autogpt-server-api/client";

type ScriptType = "full" | "e2e";

export function GenerateTestDataButton() {
  const { toast } = useToast();
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [scriptType, setScriptType] = useState<ScriptType>("e2e");
  const [result, setResult] = useState<{
    success: boolean;
    message: string;
    details?: Record<string, unknown>;
  } | null>(null);

  const handleGenerate = async () => {
    setIsSubmitting(true);
    setResult(null);

    try {
      const api = new BackendAPI();
      const response = await api.generateTestData(scriptType);

      setResult({
        success: response.success,
        message: response.message,
        details: response.details,
      });

      if (response.success) {
        toast({
          title: "Success",
          description: response.message,
        });
      } else {
        toast({
          title: "Error",
          description: response.message,
          variant: "destructive",
        });
      }
    } catch (error) {
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
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <>
      <Button
        size="lg"
        variant="default"
        onClick={() => {
          setIsDialogOpen(true);
          setResult(null);
        }}
      >
        Generate Test Data
      </Button>

      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Generate Test Data</DialogTitle>
            <DialogDescription className="pt-2">
              This will populate the database with sample test data including
              users, agents, graphs, store listings, and more.
            </DialogDescription>
          </DialogHeader>

          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="scriptType">Script Type</Label>
              <Select
                value={scriptType}
                onValueChange={(value) => setScriptType(value as ScriptType)}
                disabled={isSubmitting}
              >
                <SelectTrigger id="scriptType">
                  <SelectValue placeholder="Select script type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="e2e">
                    <div className="flex flex-col">
                      <span className="font-medium">E2E Test Data</span>
                      <span className="text-xs text-gray-500">
                        15 users with graphs, agents, and store submissions
                      </span>
                    </div>
                  </SelectItem>
                  <SelectItem value="full">
                    <div className="flex flex-col">
                      <span className="font-medium">Full Test Data</span>
                      <span className="text-xs text-gray-500">
                        100+ users with comprehensive data (takes longer)
                      </span>
                    </div>
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="rounded-md bg-yellow-50 p-3 text-sm text-yellow-800">
              <strong>Warning:</strong> This will add significant data to your
              database. Use with caution in production environments.
            </div>

            {result && (
              <div
                className={`rounded-md p-3 text-sm ${
                  result.success
                    ? "bg-green-50 text-green-800"
                    : "bg-red-50 text-red-800"
                }`}
              >
                <p className="font-medium">{result.message}</p>
                {result.details && (
                  <ul className="mt-2 list-inside list-disc">
                    {Object.entries(result.details).map(([key, value]) => (
                      <li key={key}>
                        {key.replace(/_/g, " ")}: {String(value)}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </div>

          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => setIsDialogOpen(false)}
              disabled={isSubmitting}
            >
              Cancel
            </Button>
            <Button
              type="button"
              onClick={handleGenerate}
              disabled={isSubmitting}
            >
              {isSubmitting ? "Generating..." : "Generate Test Data"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
