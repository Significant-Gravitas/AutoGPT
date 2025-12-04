"use client";

import { useState } from "react";
import { WidgetProps } from "@rjsf/utils";
import {
  InputType,
  mapJsonSchemaTypeToInputType,
} from "@/app/(platform)/build/components/FlowEditor/nodes/helpers";
import { Input } from "@/components/atoms/Input/Input";
import { Button } from "@/components/atoms/Button/Button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { BlockUIType } from "@/lib/autogpt-server-api/types";
import { InputExpanderModal } from "./InputExpanderModal";
import { ArrowsOutIcon } from "@phosphor-icons/react";

export const TextInputWidget = (props: WidgetProps) => {
  const { schema, formContext } = props;
  const { uiType, size = "small" } = formContext as {
    uiType: BlockUIType;
    size?: string;
  };

  const [isModalOpen, setIsModalOpen] = useState(false);

  const mapped = mapJsonSchemaTypeToInputType(schema);

  type InputConfig = {
    htmlType: string;
    placeholder: string;
    handleChange: (v: string) => any;
  };

  const inputConfig: Partial<Record<InputType, InputConfig>> = {
    [InputType.TEXT_AREA]: {
      htmlType: "textarea",
      placeholder: "Enter text...",
      handleChange: (v: string) => (v === "" ? undefined : v),
    },
    [InputType.PASSWORD]: {
      htmlType: "password",
      placeholder: "Enter secret text...",
      handleChange: (v: string) => (v === "" ? undefined : v),
    },
    [InputType.NUMBER]: {
      htmlType: "number",
      placeholder: "Enter number value...",
      handleChange: (v: string) => (v === "" ? undefined : Number(v)),
    },
    [InputType.INTEGER]: {
      htmlType: "account",
      placeholder: "Enter integer value...",
      handleChange: (v: string) => (v === "" ? undefined : Number(v)),
    },
  };

  const defaultConfig: InputConfig = {
    htmlType: "text",
    placeholder: "Enter string value...",
    handleChange: (v: string) => (v === "" ? undefined : v),
  };

  const config = (mapped && inputConfig[mapped]) || defaultConfig;

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) => {
    const v = e.target.value;
    return props.onChange(config.handleChange(v));
  };

  const handleModalSave = (value: string) => {
    props.onChange(config.handleChange(value));
    setIsModalOpen(false);
  };

  const handleModalOpen = () => {
    setIsModalOpen(true);
  };

  // Determine input size based on context
  const inputSize = size === "large" ? "medium" : "small";

  // Check if this input type should show the expand button
  // Show for text and password types, not for number/integer
  const showExpandButton =
    config.htmlType === "text" ||
    config.htmlType === "password" ||
    config.htmlType === "textarea";

  if (uiType === BlockUIType.NOTE) {
    return (
      <Input
        id={props.id}
        hideLabel={true}
        type={"textarea"}
        label={""}
        size="small"
        wrapperClassName="mb-0"
        value={props.value ?? ""}
        className="!h-[230px] resize-none rounded-none border-none bg-transparent p-0 placeholder:text-black/60 focus:ring-0"
        onChange={handleChange}
        placeholder={"Write your note here..."}
        required={props.required}
        disabled={props.disabled}
      />
    );
  }

  return (
    <>
      <div className="nodrag relative flex items-center gap-2">
        <Input
          id={props.id}
          hideLabel={true}
          type={config.htmlType as any}
          label={""}
          size={inputSize as any}
          wrapperClassName="mb-0 flex-1"
          value={props.value ?? ""}
          onChange={handleChange}
          placeholder={schema.placeholder || config.placeholder}
          required={props.required}
          disabled={props.disabled}
          className={showExpandButton ? "pr-8" : ""}
        />
        {showExpandButton && (
          <Tooltip delayDuration={0}>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                onClick={handleModalOpen}
                type="button"
                className="p-1"
              >
                <ArrowsOutIcon className="size-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Expand input</TooltipContent>
          </Tooltip>
        )}
      </div>

      <InputExpanderModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSave={handleModalSave}
        title={schema.title || "Edit value"}
        description={schema.description || ""}
        defaultValue={props.value ?? ""}
        placeholder={schema.placeholder || config.placeholder}
      />
    </>
  );
};
