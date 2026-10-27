import React from "react";
import { getErrorMessage, getHttpErrorMessage } from "./helpers";
import { CardWrapper } from "./components/CardWrapper";
import { ErrorHeader } from "./components/ErrorHeader";
import { ErrorMessage } from "./components/ErrorMessage";
import { ActionButtons } from "./components/ActionButtons";

export interface ErrorCardProps {
  isSuccess?: boolean;
  isOurProblem?: boolean;
  responseError?: {
    detail?: Array<{ msg: string }> | string;
    message?: string;
  };
  httpError?: {
    status?: number;
    statusText?: string;
    message?: string;
  };
  context?: string;
  hint?: string;
  onRetry?: () => void;
  className?: string;
  /** `body-md` inside a profile tab, whose copy reads at 15px. */
  textSize?: "body" | "body-md";
}

export function ErrorCard({
  isSuccess = false,
  isOurProblem = true,
  responseError,
  httpError,
  context = "data",
  hint,
  onRetry,
  className = "",
  textSize = "body",
}: ErrorCardProps) {
  if (isSuccess && !responseError && !httpError) {
    return null;
  }

  const hasResponseDetail = !!(
    responseError &&
    ((typeof responseError.detail === "string" &&
      responseError.detail.length > 0) ||
      (Array.isArray(responseError.detail) &&
        responseError.detail.length > 0) ||
      (responseError.message && responseError.message.length > 0))
  );

  const errorMessage = hasResponseDetail
    ? getErrorMessage(responseError)
    : getHttpErrorMessage(httpError);

  return (
    <CardWrapper className={className}>
      <div data-slot="error-card-content" className="relative space-y-4 p-6">
        <ErrorHeader />
        <ErrorMessage
          errorMessage={errorMessage}
          context={context}
          hint={hint}
          textSize={textSize}
        />
        {isOurProblem && (
          <ActionButtons
            onRetry={onRetry}
            responseError={responseError}
            httpError={httpError}
            context={context}
            textSize={textSize}
          />
        )}
      </div>
    </CardWrapper>
  );
}
