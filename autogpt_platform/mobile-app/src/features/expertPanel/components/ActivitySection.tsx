import { Children, Fragment, type ReactNode } from "react";
import { StyleSheet, Text, View } from "react-native";

import { createStyles, spacing } from "@/lib/theme";

interface Props {
  title: string;
  isLoading: boolean;
  children: ReactNode;
}

export function ActivitySection({ title, isLoading, children }: Props) {
  const styles = useStyles();
  const rows = Children.toArray(children);

  return (
    <View style={styles.section}>
      <Text accessibilityRole="header" style={styles.title}>
        {title}
      </Text>
      <View style={styles.card}>
        {isLoading ? (
          <View
            accessibilityLabel={`Loading ${title.toLowerCase()}`}
            style={styles.placeholder}
          >
            <View style={styles.placeholderLine} />
          </View>
        ) : (
          rows.map((row, index) => (
            <Fragment key={index}>
              {index > 0 ? <View style={styles.hairline} /> : null}
              {row}
            </Fragment>
          ))
        )}
      </View>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  section: { gap: spacing.sm },
  title: {
    fontSize: 13,
    lineHeight: 18,
    color: colors.textTertiary,
    paddingHorizontal: spacing.md,
  },
  card: {
    backgroundColor: colors.surface,
    borderWidth: 0.5,
    borderColor: colors.border,
    borderRadius: 16,
    overflow: "hidden",
  },
  hairline: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: colors.border,
    marginLeft: spacing.md,
  },
  placeholder: { paddingHorizontal: spacing.md, paddingVertical: spacing.lg },
  placeholderLine: {
    height: 14,
    width: "60%",
    borderRadius: 7,
    backgroundColor: colors.surfacePressed,
  },
}));
