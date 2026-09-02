import React from "react";
import { handleReportError } from "../helpers";
import { ErrorCardProps } from "../ErrorCard";
import { Button } from "@/components/atoms/Button/Button";
import {
  Bug01Icon,
  DiscordIcon,
  Refresh01Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface ActionButtonsProps {
  onRetry?: () => void;
  responseError?: ErrorCardProps["responseError"];
  httpError?: ErrorCardProps["httpError"];
  context: string;
}

export function ActionButtons({
  onRetry,
  responseError,
  httpError,
  context,
}: ActionButtonsProps) {
  return (
    <div className="flex flex-col flex-wrap gap-3 pt-2 sm:flex-row">
      {onRetry && (
        <Button onClick={onRetry} variant="outline" size="small">
          <Icon icon={Refresh01Icon} size={16} />
          Try Again
        </Button>
      )}

      <Button
        onClick={() => handleReportError(responseError, httpError, context)}
        variant="ghost"
        size="small"
      >
        <Icon icon={Bug01Icon} size={16} />
        Report Error
      </Button>

      <Button
        as="NextLink"
        variant="ghost"
        size="small"
        href="https://discord.gg/autogpt"
        target="_blank"
        rel="noopener noreferrer"
      >
        <Icon icon={DiscordIcon} size={16} />
        Get Help
      </Button>
    </div>
  );
}
