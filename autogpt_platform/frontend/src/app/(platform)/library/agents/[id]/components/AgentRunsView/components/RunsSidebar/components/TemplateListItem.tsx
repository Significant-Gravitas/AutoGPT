"use client";

import React from "react";
import { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";
import { RunSidebarCard } from "./RunSidebarCard";
import { IconWrapper } from "./RunIconWrapper";
import { LinkIcon, PushPinIcon } from "@phosphor-icons/react";

interface TemplateListItemProps {
  preset: LibraryAgentPreset;
  selected?: boolean;
  onClick?: () => void;
}

export function TemplateListItem({
  preset,
  selected,
  onClick,
}: TemplateListItemProps) {
  const isTrigger = !!preset.webhook;
  const isActive = preset.is_active ?? false;

  return (
    <RunSidebarCard
      title={preset.name}
      description={preset.description || "No description"}
      onClick={onClick}
      selected={selected}
      icon={
        <IconWrapper
          className={
            isTrigger
              ? isActive
                ? "border-green-50 bg-green-50"
                : "border-gray-50 bg-gray-50"
              : "border-blue-50 bg-blue-50"
          }
        >
          {isTrigger ? (
            <LinkIcon
              size={16}
              className={isActive ? "text-green-700" : "text-gray-700"}
              weight="bold"
            />
          ) : (
            <PushPinIcon size={16} className="text-blue-700" weight="bold" />
          )}
        </IconWrapper>
      }
      statusBadge={
        isTrigger ? (
          <span
            className={`rounded-full px-2 py-0.5 text-xs ${
              isActive
                ? "bg-green-100 text-green-800"
                : "bg-gray-100 text-gray-600"
            }`}
          >
            {isActive ? "Active" : "Inactive"}
          </span>
        ) : (
          <span className="rounded-full bg-blue-100 px-2 py-0.5 text-xs text-blue-800">
            Preset
          </span>
        )
      }
    />
  );
}
