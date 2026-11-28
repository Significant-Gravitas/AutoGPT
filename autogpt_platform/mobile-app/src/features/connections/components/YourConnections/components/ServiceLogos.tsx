import { Image } from "expo-image";
import { View } from "react-native";

import { createStyles } from "@/lib/theme";

import { logoURL } from "../../../helpers";

// Common services, to show what connecting means before anything is.
const LOGO_IDS = ["google", "slack", "notion", "github", "linear"];
const SIZE = 42;

/** A row of overlapping round logos. */
export function ServiceLogos() {
  const styles = useStyles();
  return (
    <View
      accessibilityElementsHidden
      importantForAccessibility="no-hide-descendants"
      style={styles.row}
    >
      {LOGO_IDS.map((id, index) => (
        <View key={id} style={[styles.logo, index > 0 && styles.overlap]}>
          <Image source={{ uri: logoURL(id) }} style={styles.image} contentFit="contain" />
        </View>
      ))}
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  row: { flexDirection: "row", justifyContent: "center" },
  logo: {
    width: SIZE,
    height: SIZE,
    borderRadius: SIZE / 2,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.panelCard,
    borderWidth: 3,
    borderColor: colors.panelCard,
    shadowColor: "#000000",
    shadowOpacity: 0.08,
    shadowRadius: 4,
    shadowOffset: { width: 0, height: 1 },
    elevation: 1,
  },
  overlap: { marginLeft: -8 },
  image: { width: 20, height: 20 },
}));
