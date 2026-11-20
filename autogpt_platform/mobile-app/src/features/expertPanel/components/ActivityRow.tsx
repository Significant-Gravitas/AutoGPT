import type { ReactNode } from "react";
import { Text, View } from "react-native";

import { createStyles, spacing, type } from "@/lib/theme";

interface Props {
  leading?: ReactNode;
  title: string;
  detail?: string | null;
  trailing?: ReactNode;
}

export function ActivityRow({ leading, title, detail, trailing }: Props) {
  const styles = useStyles();

  return (
    <View style={styles.row}>
      {leading ? <View style={styles.leading}>{leading}</View> : null}
      <View style={styles.text}>
        <Text numberOfLines={1} style={styles.title}>
          {title}
        </Text>
        {detail ? (
          <Text numberOfLines={1} style={styles.detail}>
            {detail}
          </Text>
        ) : null}
      </View>
      {trailing}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    minHeight: 60,
  },
  leading: { width: 20, alignItems: "center" },
  text: { flex: 1, gap: 2 },
  title: { ...type.label, color: colors.text },
  detail: { ...type.small, color: colors.textTertiary },
}));
