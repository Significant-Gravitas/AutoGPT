import { View } from "react-native";
import Svg, { Defs, LinearGradient, Rect, Stop } from "react-native-svg";

import { createStyles, useColors } from "@/lib/theme";

interface Props {
  color: string;
  height: number;
  /** Held solid this far down, under the header's fade, so the two meet without a seam. */
  solidHeight: number;
}

/** The profile's coloured top, fading down into the page grey. */
export function AreaWash({ color, height, solidHeight }: Props) {
  const styles = useStyles();
  const colors = useColors();
  return (
    <View style={[styles.wash, { height }]}>
      <Svg width="100%" height="100%" preserveAspectRatio="none">
        <Defs>
          <LinearGradient id="areaWash" x1="0" y1="0" x2="0" y2="1">
            <Stop offset="0" stopColor={color} />
            <Stop offset={Math.min(solidHeight / height, 0.9)} stopColor={color} />
            <Stop offset="1" stopColor={colors.panelBackground} />
          </LinearGradient>
        </Defs>
        <Rect width="100%" height="100%" fill="url(#areaWash)" />
      </Svg>
    </View>
  );
}

const useStyles = createStyles(() => ({
  wash: { position: "absolute", top: 0, left: 0, right: 0, pointerEvents: "none" },
}));
