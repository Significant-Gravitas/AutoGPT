"use client";

import {
  File02Icon,
  Image01Icon,
  Table01Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { PanelEmptyState } from "../../ContextPanel/components/PanelEmptyState/PanelEmptyState";

// Only Otto's chats reach this: an expert's chat always lists that expert's
// documents instead.
export function FilesEmptyState() {
  return (
    <PanelEmptyState
      tiles={[
        <Icon
          key="image"
          icon={Image01Icon}
          size={22}
          // TODO(tokens): no exact role for text-sky-500
          className="text-sky-500"
        />,
        <Icon
          key="table"
          icon={Table01Icon}
          size={22}
          // TODO(tokens): no exact role for text-emerald-500
          className="text-emerald-500"
        />,
        <Icon
          key="doc"
          icon={File02Icon}
          size={22}
          // TODO(tokens): no exact role for text-violet-500
          className="text-violet-500"
        />,
      ]}
      title="No files yet"
      description="Files you attach and documents Otto creates in this chat will show up here."
    />
  );
}
