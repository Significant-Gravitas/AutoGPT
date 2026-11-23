import type { ReactNode } from "react";
import { Text, View } from "react-native";

import { createStyles, spacing } from "@/lib/theme";

interface Props {
  title: string;
  count?: number;
  description?: string;
  children: ReactNode;
}

/** One block of the profile: a heading, an optional line under it, then its content. */
export function ProfileSection({ title, count, description, children }: Props) {
  const styles = useStyles();
  return (
    <View>
      <View style={styles.heading}>
        <View style={styles.titleRow}>
          <Text accessibilityRole="header" style={styles.title}>
            {title}
          </Text>
          {count ? <Text style={styles.count}>{count}</Text> : null}
        </View>
        {description ? <Text style={styles.description}>{description}</Text> : null}
      </View>
      {children}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  heading: { paddingHorizontal: spacing.xs, paddingBottom: 10, gap: 3 },
  titleRow: { flexDirection: "row", alignItems: "baseline", gap: spacing.sm },
  title: { fontSize: 19, lineHeight: 24, fontWeight: "700", letterSpacing: -0.2, color: colors.text },
  count: { fontSize: 14, lineHeight: 18, fontWeight: "500", color: colors.textTertiary },
  description: { fontSize: 14, lineHeight: 19, color: colors.textSecondary },
}));
