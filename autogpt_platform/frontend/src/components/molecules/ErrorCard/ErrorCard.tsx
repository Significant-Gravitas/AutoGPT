import React from "react";
import { getErrorMessage, getHttpErrorMessage } from "./helpers";
import { CardWrapper } from "./components/CardWrapper";
import { ErrorHeader } from "./components/ErrorHeader";
import { ErrorMessage } from "./components/ErrorMessage";
import { ActionButtons } from "./components/ActionButtons";

export interface ErrorCardProps {
  isSuccess?: boolean;
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
  onRetry?: () => void;
  className?: string;
}

export function ErrorCard({
  isSuccess = false,
  responseError,
  httpError,
  context = "data",
  onRetry,
  className = "",
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
      <div className="relative space-y-4 p-6">
        <ErrorHeader />
        <ErrorMessage errorMessage={errorMessage} context={context} />
        <ActionButtons
          onRetry={onRetry}
          responseError={responseError}
          httpError={httpError}
          context={context}
        />
      </div>
    </CardWrapper>
  );
}
