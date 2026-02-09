"use client";

import { useState } from "react";
import { UserMinus, UserCheck, CreditCard } from "@phosphor-icons/react";
import { Card } from "@/components/atoms/Card/Card";
import { Input } from "@/components/atoms/Input/Input";
import { Button } from "@/components/atoms/Button/Button";
import { Alert, AlertDescription } from "@/components/molecules/Alert/Alert";
import { useAdminImpersonation } from "./useAdminImpersonation";
import { useGetV1GetUserCredits } from "@/app/api/__generated__/endpoints/credits/credits";

export function AdminImpersonationPanel() {
  const [userIdInput, setUserIdInput] = useState("");
  const [error, setError] = useState("");
  const {
    isImpersonating,
    impersonatedUserId,
    startImpersonating,
    stopImpersonating,
  } = useAdminImpersonation();

  // Demo: Use existing credits API - it will automatically use impersonation if active
  const {
    data: creditsResponse,
    isLoading: creditsLoading,
    error: creditsError,
  } = useGetV1GetUserCredits();

  function handleStartImpersonation() {
    setError("");

    if (!userIdInput.trim()) {
      setError("Please enter a valid user ID");
      return;
    }

    // Basic UUID validation
    const uuidRegex =
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
    if (!uuidRegex.test(userIdInput.trim())) {
      setError("Please enter a valid UUID format user ID");
      return;
    }

    try {
      startImpersonating(userIdInput.trim());
      setUserIdInput("");
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to start impersonation",
      );
    }
  }

  function handleStopImpersonation() {
    stopImpersonating();
    setError("");
  }

  return (
    <Card className="w-full max-w-2xl">
      <div className="space-y-4">
        <div className="border-b pb-4">
          <div className="mb-2 flex items-center space-x-2">
            <UserCheck className="h-5 w-5" />
            <h2 className="text-xl font-semibold">Admin User Impersonation</h2>
          </div>
          <p className="text-sm text-gray-600">
            Act on behalf of another user for debugging and support purposes
          </p>
        </div>

        {/* Security Warning */}
        <Alert variant="error">
          <AlertDescription>
            <strong>Security Notice:</strong> This feature is for admin
            debugging and support only. All impersonation actions are logged for
            audit purposes.
          </AlertDescription>
        </Alert>

        {/* Current Status */}
        {isImpersonating && (
          <Alert variant="warning">
            <AlertDescription>
              <strong>Currently impersonating:</strong>{" "}
              <code className="rounded bg-amber-100 px-1 font-mono text-sm">
                {impersonatedUserId}
              </code>
            </AlertDescription>
          </Alert>
        )}

        {/* Impersonation Controls */}
        <div className="space-y-3">
          <Input
            label="User ID to Impersonate"
            id="user-id-input"
            placeholder="e.g., 2e7ea138-2097-425d-9cad-c660f29cc251"
            value={userIdInput}
            onChange={(e) => setUserIdInput(e.target.value)}
            disabled={isImpersonating}
            error={error}
          />

          <div className="flex space-x-2">
            <Button
              onClick={handleStartImpersonation}
              disabled={isImpersonating || !userIdInput.trim()}
              className="min-w-[100px]"
            >
              {isImpersonating ? "Active" : "Start"}
            </Button>

            {isImpersonating && (
              <Button
                onClick={handleStopImpersonation}
                variant="secondary"
                leftIcon={<UserMinus className="h-4 w-4" />}
              >
                Stop Impersonation
              </Button>
            )}
          </div>
        </div>

        {/* Demo: Live Credits Display */}
        <Card className="bg-gray-50">
          <div className="space-y-3">
            <div className="flex items-center space-x-2">
              <CreditCard className="h-4 w-4" />
              <h3 className="text-sm font-medium">Live Demo: User Credits</h3>
            </div>

            {creditsLoading ? (
              <p className="text-sm text-gray-600">Loading credits...</p>
            ) : creditsError ? (
              <Alert variant="error">
                <AlertDescription className="text-sm">
                  Error loading credits:{" "}
                  {creditsError &&
                  typeof creditsError === "object" &&
                  "message" in creditsError
                    ? String(creditsError.message)
                    : "Unknown error"}
                </AlertDescription>
              </Alert>
            ) : creditsResponse?.data ? (
              <div className="space-y-1">
                <p className="text-sm">
                  <strong>
                    {creditsResponse.data &&
                    typeof creditsResponse.data === "object" &&
                    "credits" in creditsResponse.data
                      ? String(creditsResponse.data.credits)
                      : "N/A"}
                  </strong>{" "}
                  credits available
                  {isImpersonating && (
                    <span className="ml-2 text-amber-600">
                      (via impersonation)
                    </span>
                  )}
                </p>
                <p className="text-xs text-gray-500">
                  {isImpersonating
                    ? `Showing credits for user ${impersonatedUserId}`
                    : "Showing your own credits"}
                </p>
              </div>
            ) : (
              <p className="text-sm text-gray-600">No credits data available</p>
            )}
          </div>
        </Card>

        {/* Instructions */}
        <div className="space-y-1 text-sm text-gray-600">
          <p>
            <strong>Instructions:</strong>
          </p>
          <ul className="ml-2 list-inside list-disc space-y-1">
            <li>Enter the UUID of the user you want to impersonate</li>
            <li>
              All existing API endpoints automatically work with impersonation
            </li>
            <li>A warning banner will appear while impersonation is active</li>
            <li>
              Impersonation persists across page refreshes in this session
            </li>
          </ul>
        </div>
      </div>
    </Card>
  );
}
