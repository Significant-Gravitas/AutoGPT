import {
  BracesIcon,
  CodeIcon,
  File02Icon,
  FileEmpty02Icon,
  Image01Icon,
  Pdf01Icon,
  Presentation01Icon,
  TableIcon,
  Video01Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react-native";
import { Image } from "expo-image";
import { View } from "react-native";

import { Icon } from "@/components/Icon";
import type { WorkspaceFile } from "@/lib/api/workspace";
import { createStyles, useColors } from "@/lib/theme";

import { fileTypeIcon, type FileTypeIcon } from "../listHelpers";
import { useFileThumbnail } from "../useFileThumbnail";

const ICONS: Record<FileTypeIcon, IconSvgElement> = {
  code: CodeIcon,
  slides: Presentation01Icon,
  document: File02Icon,
  table: TableIcon,
  image: Image01Icon,
  video: Video01Icon,
  pdf: Pdf01Icon,
  json: BracesIcon,
  other: FileEmpty02Icon,
};

interface Props {
  file: WorkspaceFile;
}

/** The web's file tile: an image's thumbnail, or an icon for its type. */
export function FileThumbnail({ file }: Props) {
  const { image, handleError } = useFileThumbnail(file);
  const styles = useStyles();
  const colors = useColors();

  return (
    <View style={styles.tile}>
      {image ? (
        <Image
          source={image}
          contentFit="cover"
          onError={handleError}
          style={styles.image}
        />
      ) : (
        <Icon icon={ICONS[fileTypeIcon(file)]} size={18} color={colors.textTertiary} />
      )}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  tile: {
    width: 40,
    height: 40,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.borderBase,
    backgroundColor: colors.background,
    alignItems: "center",
    justifyContent: "center",
    overflow: "hidden",
  },
  image: { width: "100%", height: "100%" },
}));
