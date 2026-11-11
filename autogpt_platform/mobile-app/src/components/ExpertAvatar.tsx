import { Image, type ImageSource } from "expo-image";
import { useState } from "react";
import { StyleSheet, Text, View } from "react-native";

import { createStyles } from "@/lib/theme";

interface Props {
  name: string;
  source: string | number | null;
  size: number;
}

export function ExpertAvatar({ name, source, size }: Props) {
  return <AvatarImage key={String(source)} name={name} source={source} size={size} />;
}

function AvatarImage({ name, source, size }: Props) {
  const styles = useStyles();
  const [hasFailed, setHasFailed] = useState(false);
  const shape = { width: size, height: size, borderRadius: size / 2 };

  function handleError() {
    setHasFailed(true);
  }

  if (source === null || hasFailed) {
    return (
      <View style={[shape, styles.fallback]}>
        <Text style={[styles.initial, { fontSize: size * 0.4 }]}>
          {name.trim().charAt(0).toUpperCase()}
        </Text>
      </View>
    );
  }
  const imageSource: ImageSource | number =
    typeof source === "string" ? { uri: source } : source;
  return (
    <Image
      source={imageSource}
      onError={handleError}
      contentFit="cover"
      style={[shape, styles.image]}
      accessibilityIgnoresInvertColors
    />
  );
}

const useStyles = createStyles((colors) => ({
  fallback: {
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.brandSoft,
  },
  initial: { fontWeight: "600", color: colors.text },
  image: {
    backgroundColor: colors.surfaceRaised,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.border,
  },
}));
