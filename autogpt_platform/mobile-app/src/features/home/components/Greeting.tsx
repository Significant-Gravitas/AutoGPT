import { Text, View } from "react-native";

import { ExpertAvatar } from "@/components/ExpertAvatar";
import { OTTO_AVATAR_URL, OTTO_NAME, OTTO_TINT } from "@/lib/otto";
import { createStyles, spacing } from "@/lib/theme";

interface Props {
  heading: string;
  subline: string;
}

export function Greeting({ heading, subline }: Props) {
  const styles = useStyles();
  return (
    <View style={styles.row}>
      <ExpertAvatar
        name={OTTO_NAME}
        source={OTTO_AVATAR_URL}
        size={52}
        tint={OTTO_TINT}
        inset
      />
      <View style={styles.text}>
        <Text accessibilityRole="header" style={styles.heading}>
          {heading}
        </Text>
        <Text style={styles.subline}>{subline}</Text>
      </View>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  row: { flexDirection: "row", alignItems: "center", gap: spacing.md },
  text: { flex: 1, gap: 2 },
  heading: { fontSize: 24, fontWeight: "600", lineHeight: 30, color: colors.text },
  subline: { fontSize: 14, lineHeight: 20, color: colors.textSecondary },
}));
