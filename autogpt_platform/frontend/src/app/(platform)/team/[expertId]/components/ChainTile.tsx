"use client";

import { ExpertWorkflowChainItem } from "@/app/api/__generated__/models/expertWorkflowChainItem";
import { ExpertWorkflowChainItemKind } from "@/app/api/__generated__/models/expertWorkflowChainItemKind";
import { Icon } from "@/components/atoms/Icon/Icon";
import { IntegrationLogo } from "@/components/molecules/IntegrationLogo/IntegrationLogo";
import { cn } from "@/lib/utils";
import {
  AiChat02Icon,
  FlashIcon,
  FileExportIcon,
  InputCursorTextIcon,
  Plug01Icon,
  Robot01Icon,
  UserCheck01Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";

const KIND_ICONS: Record<
  ExpertWorkflowChainItemKind,
  { icon: IconSvgElement; label: string }
> = {
  integration: { icon: Plug01Icon, label: "Integration" },
  input: { icon: InputCursorTextIcon, label: "Agent input" },
  output: { icon: FileExportIcon, label: "Agent output" },
  trigger: { icon: FlashIcon, label: "Trigger" },
  agent: { icon: Robot01Icon, label: "Agent" },
  ai: { icon: AiChat02Icon, label: "AI model" },
  mcp: { icon: Plug01Icon, label: "MCP tool" },
  human: { icon: UserCheck01Icon, label: "Human review" },
};

interface Props {
  item: ExpertWorkflowChainItem;
  size: "sm" | "md";
}

export function ChainTile({ item, size }: Props) {
  const fallback = KIND_ICONS[item.kind];
  const isSmall = size === "sm";
  return (
    <span
      className={cn(
        "flex items-center justify-center bg-white ring-1 ring-zinc-200/70",
        isSmall ? "size-9 rounded-lg" : "size-12 rounded-xl",
      )}
    >
      {item.provider ? (
        <IntegrationLogo provider={item.provider} size={isSmall ? 16 : 22} />
      ) : (
        <Icon
          icon={fallback.icon}
          size={isSmall ? 16 : 20}
          role="img"
          aria-label={fallback.label}
          className="text-zinc-600"
        />
      )}
    </span>
  );
}
