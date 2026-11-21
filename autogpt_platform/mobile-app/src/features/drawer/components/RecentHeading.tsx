import { Text } from "react-native";

import { createStyles, spacing, type } from "@/lib/theme";

import { ROW_INSET } from "../DrawerRow";

export function RecentHeading() {
  const styles = useStyles();
  return (
    <Text accessibilityRole="header" style={styles.heading}>
      Recent
    </Text>
  );
}

const useStyles = createStyles((colors) => ({
  heading: {
    ...type.label,
    fontSize: 16,
    fontWeight: "600",
    color: colors.text,
    paddingHorizontal: ROW_INSET,
    paddingTop: spacing.xl,
    paddingBottom: spacing.sm,
  },
}));
