import { Children, Fragment, type ReactNode } from "react";
import { StyleSheet, Text, View } from "react-native";

import { createStyles, radius, spacing } from "@/lib/theme";

interface Props {
  title: string;
  children: ReactNode;
}

export function SheetSection({ title, children }: Props) {
  const styles = useStyles();
  const rows = Children.toArray(children);
  return (
    <View style={styles.section}>
      <Text accessibilityRole="header" style={styles.title}>
        {title}
      </Text>
      <View style={styles.list}>
        {rows.map((row, index) => (
          <Fragment key={index}>
            {index > 0 ? <View style={styles.hairline} /> : null}
            {row}
          </Fragment>
        ))}
      </View>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  section: { gap: spacing.sm },
  title: {
    fontSize: 12,
    lineHeight: 16,
    fontWeight: "500",
    letterSpacing: 0.6,
    textTransform: "uppercase",
    color: colors.textTertiary,
  },
  list: {
    overflow: "hidden",
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.borderBase,
    backgroundColor: colors.background,
  },
  hairline: { height: StyleSheet.hairlineWidth, backgroundColor: colors.border },
}));
