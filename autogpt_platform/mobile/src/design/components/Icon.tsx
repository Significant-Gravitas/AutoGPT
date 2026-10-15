import { SymbolView, type AndroidSymbol, type SFSymbol } from "expo-symbols";
import type { ColorValue, StyleProp, ViewStyle } from "react-native";

import { colors } from "../theme";

interface Props {
  name: SFSymbol;
  /** Material Symbol for the Android follow-up; iOS renders SF Symbols. */
  android?: AndroidSymbol;
  size?: number;
  color?: ColorValue;
  weight?: "regular" | "medium" | "semibold" | "bold";
  style?: StyleProp<ViewStyle>;
}

/** Decorative system symbol. Give the surrounding control its accessibility label. */
export function Icon({
  name,
  android,
  size = 20,
  color = colors.secondaryLabel,
  weight,
  style,
}: Props) {
  return (
    <SymbolView
      name={android ? { ios: name, android, web: android } : name}
      size={size}
      tintColor={color}
      weight={weight}
      style={[{ width: size, height: size }, style]}
      accessibilityElementsHidden
      importantForAccessibility="no-hide-descendants"
    />
  );
}
