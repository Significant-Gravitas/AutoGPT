import { Text, View } from "react-native";

import { createStyles } from "@/lib/theme";

interface Props {
  title: string;
  count?: number | null;
}

export function SectionHeading({ title, count }: Props) {
  const styles = useStyles();
  return (
    <View style={styles.heading}>
      <Text accessibilityRole="header" style={styles.title}>
        {title}
      </Text>
      {count ? <Text style={styles.count}>{count}</Text> : null}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  heading: {
    flexDirection: "row",
    alignItems: "baseline",
    justifyContent: "space-between",
    paddingHorizontal: 18,
    paddingBottom: 6,
  },
  title: { fontSize: 14.5, lineHeight: 19, fontWeight: "600", color: colors.text },
  count: { fontSize: 11, lineHeight: 15, color: colors.textTertiary },
}));
