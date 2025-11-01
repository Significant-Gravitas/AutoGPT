import React from "react";
import { FieldTemplateProps } from "@rjsf/utils";
import { InfoIcon } from "@phosphor-icons/react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

const FieldTemplate: React.FC<FieldTemplateProps> = ({
  id: fieldId,
  label,
  required,
  description,
  children,
  schema,
}) => {
  return (
    <div className="w-[350px] space-y-1 pt-4">
      {label && schema.type && (
        <label htmlFor={fieldId} className="flex items-center gap-1">
          <Text variant="body" className={cn("line-clamp-1")}>
            {label}
          </Text>

          {required && <span style={{ color: "red" }}>*</span>}
          {description?.props?.description && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span
                    style={{ marginLeft: 6, cursor: "pointer" }}
                    aria-label="info"
                    tabIndex={0}
                  >
                    <InfoIcon />
                  </span>
                </TooltipTrigger>
                <TooltipContent>{description}</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </label>
      )}
      {<div className="pl-2">{children}</div>}{" "}
    </div>
  );
};

export default FieldTemplate;
