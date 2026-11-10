import { StyleSheet, Text, View } from "react-native";

import { colors, type } from "@/lib/theme";

export default function HomeScreen() {
  return (
    <View style={styles.screen}>
      <Text style={styles.text}>Signed in.</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.background,
  },
  text: { ...type.body, color: colors.textSecondary },
});
