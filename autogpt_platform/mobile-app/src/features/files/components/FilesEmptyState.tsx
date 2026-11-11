import { File02Icon, Image01Icon, Table01Icon } from "@hugeicons/core-free-icons";

import { Icon } from "@/components/Icon";
import { PanelEmptyState } from "@/components/PanelEmptyState";
import { useColors } from "@/lib/theme";

// Only Otto's chats reach this: an expert's chat always lists that expert's
// documents instead.
export function FilesEmptyState() {
  const colors = useColors();
  return (
    <PanelEmptyState
      tiles={[
        <Icon key="image" icon={Image01Icon} size={22} color={colors.sky} />,
        <Icon key="table" icon={Table01Icon} size={22} color={colors.success} />,
        <Icon key="doc" icon={File02Icon} size={22} color={colors.brand} />,
      ]}
      title="No files yet"
      description="Files you attach and documents Otto creates in this chat will show up here."
    />
  );
}
