import { Image } from "expo-image";
import { useState } from "react";
import { StyleSheet, Text, View } from "react-native";

import { colors } from "../theme";

interface Props {
  name: string;
  /** An absolute https URL, or a site path resolved against the web origin. */
  uri: string | null;
  size?: number;
}

/** A participant's portrait, falling back to their initial when there is no usable image. */
export function Avatar({ name, uri, size = 44 }: Props) {
  const [failed, setFailed] = useState(false);
  const frame = { width: size, height: size, borderRadius: size / 2 };
  if (!uri || failed) {
    return (
      <View style={[styles.fallback, frame]} accessible={false}>
        <Text
          style={[styles.initial, { fontSize: size * 0.42 }]}
          allowFontScaling={false}
        >
          {name.trim().charAt(0).toUpperCase() || "?"}
        </Text>
      </View>
    );
  }
  return (
    <Image
      source={{ uri }}
      style={[styles.image, frame]}
      contentFit="cover"
      transition={150}
      cachePolicy="memory-disk"
      recyclingKey={uri}
      accessible={false}
      onError={() => setFailed(true)}
    />
  );
}

/**
 * Avatar paths such as `/autogpt-characters/...` are public assets of the
 * selected server's web app, which also serves its auth routes.
 */
export function resolveAvatarURL(
  path: string | null,
  webOrigin: string,
): string | null {
  if (!path) return null;
  if (path.startsWith("https://")) return path;
  if (path.startsWith("/") && !path.startsWith("//"))
    return `${webOrigin}${path}`;
  return null;
}

const styles = StyleSheet.create({
  fallback: {
    backgroundColor: colors.fill,
    alignItems: "center",
    justifyContent: "center",
  },
  initial: { color: colors.secondaryLabel, fontWeight: "600" },
  image: { backgroundColor: colors.fill },
});
