import { Image, type ImageSource } from "expo-image";
import { useState } from "react";
import { Text, View } from "react-native";

import { mixHex, palette } from "@/lib/expertPalette";
import {
  avatarTintPercent,
  createStyles,
  useColors,
  useScheme,
} from "@/lib/theme";

interface Props {
  name: string;
  source: string | number | null;
  size: number;
  /** The expert's palette hex; the circle behind the picture is a soft mix of it. */
  tint?: string | null;
  /** Draw the picture at 70% of the circle, for art with its own margin. */
  inset?: boolean;
}

export function ExpertAvatar(props: Props) {
  return <AvatarImage key={String(props.source)} {...props} />;
}

function AvatarImage({ name, source, size, tint, inset }: Props) {
  const styles = useStyles();
  const colors = useColors();
  const scheme = useScheme();
  const [hasFailed, setHasFailed] = useState(false);
  const circle = {
    width: size,
    height: size,
    borderRadius: size / 2,
    backgroundColor: mixHex(
      tint ?? palette.general.hex,
      avatarTintPercent[scheme],
      colors.avatarTintBase,
    ),
  };

  function handleError() {
    setHasFailed(true);
  }

  if (source === null || hasFailed) {
    return (
      <View style={[styles.circle, circle]}>
        <Text style={[styles.initial, { fontSize: size * 0.4 }]}>
          {name.trim().charAt(0).toUpperCase()}
        </Text>
      </View>
    );
  }
  const imageSource: ImageSource | number =
    typeof source === "string" ? { uri: source } : source;
  const inner = inset ? size * 0.7 : size;
  return (
    <View style={[styles.circle, circle]}>
      <Image
        source={imageSource}
        onError={handleError}
        contentFit={inset ? "contain" : "cover"}
        style={{ width: inner, height: inner }}
        accessibilityIgnoresInvertColors
      />
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  circle: {
    alignItems: "center",
    justifyContent: "center",
    overflow: "hidden",
    borderWidth: 1,
    borderColor: colors.avatarBorder,
  },
  initial: { fontWeight: "600", color: colors.text },
}));
