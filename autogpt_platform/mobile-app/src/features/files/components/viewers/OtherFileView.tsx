import { LinkSquare02Icon } from "@hugeicons/core-free-icons";
import { Text, View } from "react-native";

import { Button } from "@/components/Button";
import type { WorkspaceFile } from "@/lib/api/workspace";
import { createStyles, radius, spacing } from "@/lib/theme";

import { FileThumbnail } from "../FileThumbnail";

interface Props {
  file: WorkspaceFile;
  isSharing: boolean;
  onOpenIn: () => void;
}

function extensionOf(name: string) {
  const dot = name.lastIndexOf(".");
  return dot > 0 ? name.slice(dot + 1).toUpperCase() : null;
}

/** A file the app can't draw: its type, and a way to open it in an app that can. */
export function OtherFileView({ file, isSharing, onOpenIn }: Props) {
  const styles = useStyles();
  const extension = extensionOf(file.name);

  return (
    <View style={styles.center}>
      <View style={styles.tile}>
        <FileThumbnail file={file} />
        {extension ? <Text style={styles.extension}>{extension}</Text> : null}
      </View>
      <Text style={styles.name}>{file.name}</Text>
      <Text style={styles.text}>
        This file doesn&apos;t open in the app yet. Open it in another app on your phone.
      </Text>
      <View style={styles.button}>
        <Button
          label="Open in…"
          onPress={onOpenIn}
          leadingIcon={LinkSquare02Icon}
          isLoading={isSharing}
          fullWidth
        />
      </View>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  center: { flex: 1, alignItems: "center", justifyContent: "center", padding: spacing.xxl, paddingBottom: 80 },
  tile: {
    width: 96,
    height: 112,
    borderRadius: radius.lg,
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.sm,
    backgroundColor: colors.panelCard,
    borderWidth: 1,
    borderColor: colors.border,
  },
  extension: { fontSize: 11, fontWeight: "700", letterSpacing: 0.5, color: colors.textSecondary },
  name: { marginTop: 20, fontSize: 18, lineHeight: 24, fontWeight: "600", color: colors.text, textAlign: "center" },
  text: { marginTop: 6, fontSize: 14.5, lineHeight: 20, color: colors.textSecondary, textAlign: "center" },
  button: { alignSelf: "stretch", marginTop: 22 },
}));
