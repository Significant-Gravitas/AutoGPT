import { File02Icon, Image01Icon, Table01Icon } from "@hugeicons/core-free-icons";

import { Icon } from "@/components/Icon";
import { PanelEmptyState } from "@/components/PanelEmptyState";
import { useColors } from "@/lib/theme";

interface Props {
  /** Set in an expert's chat, where the list is that expert's documents. */
  expertName?: string;
}

// The web's two wordings: the chat panel's for Otto, the expert page's for
// an expert with nothing yet.
export function FilesEmptyState({ expertName }: Props) {
  const colors = useColors();
  return (
    <PanelEmptyState
      tiles={[
        <Icon key="image" icon={Image01Icon} size={22} color={colors.sky} />,
        <Icon key="table" icon={Table01Icon} size={22} color={colors.success} />,
        <Icon key="doc" icon={File02Icon} size={22} color={colors.brand} />,
      ]}
      title="No files yet"
      description={
        expertName
          ? `Files you share with ${expertName} and documents it creates show up here.`
          : "Files you attach and documents Otto creates in this chat will show up here."
      }
    />
  );
}
