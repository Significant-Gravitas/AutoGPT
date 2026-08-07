"use client";

import { useState } from "react";
import {
  AiEditingIcon,
  AlertDiamondIcon,
  BookBookmarkIcon,
  BookOpenIcon,
  BrainIcon,
  BubbleChatIcon,
  CheckListIcon,
  ClockIcon,
  ComputerIcon,
  ConnectIcon,
  DatabaseIcon,
  Delete02Icon,
  FileIcon,
  FilesIcon,
  FlashIcon,
  FolderIcon,
  Globe02Icon,
  HierarchyIcon,
  InformationCircleIcon,
  MegaphoneIcon,
  MessageQuestionIcon,
  PencilEdit01Icon,
  Plug01Icon,
  PuzzleIcon,
  RefreshIcon,
  Robot01Icon,
  Search01Icon,
  Settings01Icon,
  SlidersHorizontalIcon,
  TerminalIcon,
  WrenchIcon,
} from "@hugeicons/core-free-icons";
import Image from "next/image";
import { Icon } from "@/components/atoms/Icon/Icon";
import type { ChainRow } from "./helpers";

interface RowIconProps {
  row: ChainRow;
}

interface ProviderIconProps extends RowIconProps {
  src: string;
}

export function RowIcon({ row }: RowIconProps) {
  if (row.state === "error") {
    return <Icon icon={AlertDiamondIcon} size={16} className="text-red-500" />;
  }
  const cls = "text-zinc-600";
  switch (row.category) {
    case "narration":
      return <Icon icon={AiEditingIcon} size={16} className={cls} />;
    case "reasoning":
      return <Icon icon={BrainIcon} size={16} className={cls} />;
    case "bash":
      return <Icon icon={TerminalIcon} size={16} className={cls} />;
    case "web":
      return <Icon icon={Globe02Icon} size={16} className={cls} />;
    case "browser":
      return <Icon icon={ComputerIcon} size={16} className={cls} />;
    case "file-read":
    case "file-write":
      return <Icon icon={FileIcon} size={16} className={cls} />;
    case "file-delete":
      return <Icon icon={Delete02Icon} size={16} className={cls} />;
    case "file-list":
      return <Icon icon={FilesIcon} size={16} className={cls} />;
    case "search":
      return <Icon icon={Search01Icon} size={16} className={cls} />;
    case "edit":
      return <Icon icon={PencilEdit01Icon} size={16} className={cls} />;
    case "todo":
      return <Icon icon={CheckListIcon} size={16} className={cls} />;
    case "compaction":
      return <Icon icon={RefreshIcon} size={16} className={cls} />;
    case "agent":
      return <Icon icon={Robot01Icon} size={16} className={cls} />;
    case "agent-build":
      return <Icon icon={WrenchIcon} size={16} className={cls} />;
    case "plan":
      return <Icon icon={HierarchyIcon} size={16} className={cls} />;
    case "block":
      return <Icon icon={PuzzleIcon} size={16} className={cls} />;
    case "memory":
      return <Icon icon={DatabaseIcon} size={16} className={cls} />;
    case "folder":
      return <Icon icon={FolderIcon} size={16} className={cls} />;
    case "schedule":
      return <Icon icon={ClockIcon} size={16} className={cls} />;
    case "trigger":
      return <Icon icon={FlashIcon} size={16} className={cls} />;
    case "preset":
      return <Icon icon={SlidersHorizontalIcon} size={16} className={cls} />;
    case "chat":
      return <Icon icon={BubbleChatIcon} size={16} className={cls} />;
    case "mcp":
      return <Icon icon={Plug01Icon} size={16} className={cls} />;
    case "docs":
      return <Icon icon={BookOpenIcon} size={16} className={cls} />;
    case "skill":
      return <Icon icon={BookBookmarkIcon} size={16} className={cls} />;
    case "integration":
      return <Icon icon={ConnectIcon} size={16} className={cls} />;
    case "feature":
      return <Icon icon={MegaphoneIcon} size={16} className={cls} />;
    case "question":
      return <Icon icon={MessageQuestionIcon} size={16} className={cls} />;
    case "info":
      return <Icon icon={InformationCircleIcon} size={16} className={cls} />;
    default:
      return <Icon icon={Settings01Icon} size={16} className={cls} />;
  }
}

export function ProviderIcon({ src, row }: ProviderIconProps) {
  const [failed, setFailed] = useState(false);
  if (failed) return <RowIcon row={row} />;
  return (
    <Image
      src={src}
      alt=""
      width={16}
      height={16}
      className="rounded-sm object-contain"
      onError={() => setFailed(true)}
    />
  );
}
