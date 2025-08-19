import React from "react";
import { getErrorMessage, getHttpErrorMessage, isHttpError } from "./helpers";
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
  loadingSlot?: React.ReactNode;
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

  const isHttp = isHttpError(httpError);

  const errorMessage = isHttp
    ? getHttpErrorMessage(httpError)
    : getErrorMessage(responseError);

  return (
    <CardWrapper className={className}>
      <div className="relative space-y-4 p-6">
        <ErrorHeader />
        <ErrorMessage
          isHttpError={isHttp}
          errorMessage={errorMessage}
          context={context}
        />
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
