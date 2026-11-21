import { ActivityIndicator, View } from "react-native";

import { createStyles, useColors } from "@/lib/theme";

/** Shown while the server is asked whether this account has a plan. */
export default function LaunchScreen() {
  const styles = useStyles();
  const colors = useColors();
  return (
    <View style={styles.screen}>
      <ActivityIndicator color={colors.textSecondary} />
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  screen: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.background,
  },
}));
