"use client";

import { useState } from "react";
import {
  ArrowsClockwiseIcon,
  BookBookmarkIcon,
  BookOpenIcon,
  BrainIcon,
  ChatCenteredDotsIcon,
  ChatCircleIcon,
  ClockIcon,
  DatabaseIcon,
  FileIcon,
  FilesIcon,
  FolderIcon,
  GearIcon,
  GlobeIcon,
  InfoIcon,
  LightningIcon,
  ListChecksIcon,
  MagnifyingGlassIcon,
  MegaphoneIcon,
  MonitorIcon,
  PencilSimpleIcon,
  PlugsConnectedIcon,
  PlugsIcon,
  PuzzlePieceIcon,
  RobotIcon,
  SlidersHorizontalIcon,
  TerminalIcon,
  TrashIcon,
  TreeStructureIcon,
  WarningDiamondIcon,
  WrenchIcon,
} from "@phosphor-icons/react";
import Image from "next/image";
import type { ChainRow } from "./helpers";

interface RowIconProps {
  row: ChainRow;
}

interface ProviderIconProps extends RowIconProps {
  src: string;
}

export function RowIcon({ row }: RowIconProps) {
  if (row.state === "error") {
    return <WarningDiamondIcon size={16} className="text-red-500" />;
  }
  const cls = "text-zinc-600";
  switch (row.category) {
    case "reasoning":
      return <BrainIcon size={16} className={cls} />;
    case "bash":
      return <TerminalIcon size={16} className={cls} />;
    case "web":
      return <GlobeIcon size={16} className={cls} />;
    case "browser":
      return <MonitorIcon size={16} className={cls} />;
    case "file-read":
    case "file-write":
      return <FileIcon size={16} className={cls} />;
    case "file-delete":
      return <TrashIcon size={16} className={cls} />;
    case "file-list":
      return <FilesIcon size={16} className={cls} />;
    case "search":
      return <MagnifyingGlassIcon size={16} className={cls} />;
    case "edit":
      return <PencilSimpleIcon size={16} className={cls} />;
    case "todo":
      return <ListChecksIcon size={16} className={cls} />;
    case "compaction":
      return <ArrowsClockwiseIcon size={16} className={cls} />;
    case "agent":
      return <RobotIcon size={16} className={cls} />;
    case "agent-build":
      return <WrenchIcon size={16} className={cls} />;
    case "plan":
      return <TreeStructureIcon size={16} className={cls} />;
    case "block":
      return <PuzzlePieceIcon size={16} className={cls} />;
    case "memory":
      return <DatabaseIcon size={16} className={cls} />;
    case "folder":
      return <FolderIcon size={16} className={cls} />;
    case "schedule":
      return <ClockIcon size={16} className={cls} />;
    case "trigger":
      return <LightningIcon size={16} className={cls} />;
    case "preset":
      return <SlidersHorizontalIcon size={16} className={cls} />;
    case "chat":
      return <ChatCircleIcon size={16} className={cls} />;
    case "mcp":
      return <PlugsIcon size={16} className={cls} />;
    case "docs":
      return <BookOpenIcon size={16} className={cls} />;
    case "skill":
      return <BookBookmarkIcon size={16} className={cls} />;
    case "integration":
      return <PlugsConnectedIcon size={16} className={cls} />;
    case "feature":
      return <MegaphoneIcon size={16} className={cls} />;
    case "question":
      return <ChatCenteredDotsIcon size={16} className={cls} />;
    case "info":
      return <InfoIcon size={16} className={cls} />;
    default:
      return <GearIcon size={16} className={cls} />;
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
