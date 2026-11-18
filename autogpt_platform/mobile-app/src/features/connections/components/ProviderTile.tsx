import { Image } from "expo-image";
import { useState } from "react";
import { Text, View } from "react-native";

import { palette } from "@/lib/expertPalette";
import { createStyles } from "@/lib/theme";

import { logoURL, pickByName, tileLetters } from "../helpers";

interface Props {
  name: string;
  /** The logo's name on the web (`logoID`); without one the tile shows initials. */
  logoID?: string | null;
  size?: number;
}

const TILE_COLOURS = Object.values(palette).map((entry) => entry.hex);

/** A service's logo, or its initials on a colour picked from its name. */
export function ProviderTile({ name, logoID, size = 32 }: Props) {
  const styles = useStyles();
  // Kept per URL, so a tile handed another service tries its logo afresh.
  const [failedURL, setFailedURL] = useState<string | null>(null);
  const frame = { width: size, height: size, borderRadius: size / 4 };
  const iconURL = logoID ? logoURL(logoID) : null;

  function handleError() {
    setFailedURL(iconURL);
  }

  if (iconURL && iconURL !== failedURL) {
    return (
      <View
        accessibilityElementsHidden
        importantForAccessibility="no-hide-descendants"
        style={[styles.tile, styles.logoTile, frame]}
      >
        <Image
          key={iconURL}
          source={{ uri: iconURL }}
          style={{ width: size * 0.7, height: size * 0.7 }}
          contentFit="contain"
          onError={handleError}
        />
      </View>
    );
  }

  return (
    <View
      accessibilityElementsHidden
      importantForAccessibility="no-hide-descendants"
      style={[styles.tile, frame, { backgroundColor: pickByName(name, TILE_COLOURS) }]}
    >
      <Text style={[styles.letters, { fontSize: size * 0.4 }]}>{tileLetters(name)}</Text>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  tile: { alignItems: "center", justifyContent: "center", overflow: "hidden" },
  logoTile: {
    borderWidth: 1,
    borderColor: colors.borderBase,
    backgroundColor: colors.surfaceRaised,
  },
  letters: { fontWeight: "700", color: "#141414" },
}));
