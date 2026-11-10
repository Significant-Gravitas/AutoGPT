import type { ReactNode } from "react";
import {
  Linking,
  StyleSheet,
  Text,
  type ImageStyle,
  type TextStyle,
} from "react-native";
import { Renderer } from "react-native-marked";

import { colors, type } from "@/lib/theme";

const WEB_URL_RE = /^https?:\/\//i;

/**
 * Replies may carry text planted by a web page, so images are never fetched
 * (their URL could carry data out) and only web links open on a tap.
 */
class SafeRenderer extends Renderer {
  override link(
    children: string | ReactNode[],
    href: string,
    styles?: TextStyle,
    title?: string,
  ): ReactNode {
    return this.safeLink(children, href, styles, title);
  }

  override image(
    _uri: string,
    alt?: string,
    _style?: ImageStyle,
    title?: string,
  ): ReactNode {
    const label = alt || title;
    if (!label) return null;
    return (
      <Text key={this.getKey()} style={safeStyles.alt}>
        {label}
      </Text>
    );
  }

  override linkImage(
    href: string,
    _imageUrl: string,
    alt?: string,
    _style?: ImageStyle,
    title?: string,
  ): ReactNode {
    return this.safeLink(alt || title || href, href, safeStyles.link, title);
  }

  private safeLink(
    children: string | ReactNode[],
    href: string,
    styles: TextStyle | undefined,
    title: string | undefined,
  ) {
    if (!WEB_URL_RE.test(href)) {
      return (
        <Text key={this.getKey()} style={[styles, safeStyles.plain]}>
          {children}
        </Text>
      );
    }
    return (
      <Text
        key={this.getKey()}
        accessibilityRole="link"
        accessibilityLabel={title || undefined}
        onPress={() => openWebLink(href)}
        style={[styles, safeStyles.link]}
      >
        {children}
      </Text>
    );
  }
}

function openWebLink(href: string) {
  Linking.openURL(href).catch(() => undefined);
}

export const safeRenderer = new SafeRenderer();

const safeStyles = StyleSheet.create({
  alt: { ...type.body, color: colors.textSecondary },
  link: { color: colors.brand },
  plain: { color: colors.text },
});
