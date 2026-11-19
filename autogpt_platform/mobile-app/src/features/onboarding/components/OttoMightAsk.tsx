import { Image } from "expo-image";
import { Text, View } from "react-native";

import { OTTO_AVATAR_URL } from "@/lib/otto";
import { createStyles, radius, spacing, type } from "@/lib/theme";

export function OttoMightAsk() {
  const styles = useStyles();
  return (
    <View style={styles.card}>
      <Image source={OTTO_AVATAR_URL} style={styles.otto} contentFit="contain" />
      <View style={styles.text}>
        <Text style={styles.label}>Otto might ask</Text>
        <Text style={styles.question}>What would you hand off first?</Text>
      </View>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  card: {
    alignSelf: "stretch",
    flexDirection: "row",
    alignItems: "flex-start",
    gap: spacing.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
  },
  otto: { width: 28, height: 28 },
  text: { flex: 1 },
  label: { ...type.small, color: colors.textTertiary },
  question: { ...type.label, color: colors.text },
}));
