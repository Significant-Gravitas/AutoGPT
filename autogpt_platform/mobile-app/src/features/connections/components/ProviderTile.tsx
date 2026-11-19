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
  const iconURL = logoID ? logoURL(logoID) : null;
  // Keyed by URL: a tile handed another logo, or its old one back, tries it afresh.
  return <Tile key={iconURL ?? ""} name={name} iconURL={iconURL} size={size} />;
}

interface TileProps {
  name: string;
  iconURL: string | null;
  size: number;
}

function Tile({ name, iconURL, size }: TileProps) {
  const styles = useStyles();
  const [hasFailed, setHasFailed] = useState(false);
  const frame = { width: size, height: size, borderRadius: size / 4 };

  function handleError() {
    setHasFailed(true);
  }

  if (iconURL && !hasFailed) {
    return (
      <View
        accessibilityElementsHidden
        importantForAccessibility="no-hide-descendants"
        style={[styles.tile, styles.logoTile, frame]}
      >
        <Image
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
