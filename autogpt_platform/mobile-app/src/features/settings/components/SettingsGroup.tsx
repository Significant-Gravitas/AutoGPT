import { Children, Fragment, type ReactNode } from "react";
import { StyleSheet, Text, View } from "react-native";

import { createStyles, radius, spacing, type } from "@/lib/theme";

interface Props {
  title?: string;
  /** A line under the card, about the whole group. */
  note?: string;
  children: ReactNode;
}

/** A white card of rows on the grey Settings page, with lines between them. */
export function SettingsGroup({ title, note, children }: Props) {
  const styles = useStyles();
  const rows = Children.toArray(children);
  return (
    <View style={styles.group}>
      {title ? <Text style={styles.title}>{title}</Text> : null}
      <View style={styles.card}>
        {rows.map((row, index) => (
          <Fragment key={index}>
            {index > 0 ? <View style={styles.divider} /> : null}
            {row}
          </Fragment>
        ))}
      </View>
      {note ? <Text style={styles.note}>{note}</Text> : null}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  group: { gap: spacing.xs },
  title: {
    ...type.small,
    marginLeft: spacing.sm,
    color: colors.textTertiary,
  },
  note: {
    ...type.small,
    marginHorizontal: spacing.sm,
    color: colors.textTertiary,
  },
  card: {
    overflow: "hidden",
    borderRadius: radius.lg,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.border,
    backgroundColor: colors.panelCard,
  },
  divider: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: colors.panelDivider,
  },
}));
