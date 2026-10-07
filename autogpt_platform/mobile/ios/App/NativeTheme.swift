import UIKit

@MainActor
enum NativeTheme {
  static let background = color(0xFEFEFE)
  static let text = color(0x141414)
  static let secondaryText = color(0x68686F)
  static let border = color(0xDADADC)
  static let primary = color(0x3E3E43)

  static func font(
    _ name: String, size: CGFloat, style: UIFont.TextStyle,
    compatibleWith traits: UITraitCollection? = nil
  ) -> UIFont {
    let base = UIFont(name: name, size: size) ?? .systemFont(ofSize: size)
    return UIFontMetrics(forTextStyle: style).scaledFont(for: base, compatibleWith: traits)
  }

  static func styleButton(_ button: UIButton, primary: Bool) {
    button.configuration = buttonConfiguration(primary: primary)
    button.configurationUpdateHandler = { button in
      let title = button.configuration?.title ?? button.title(for: .normal)
      var configuration = buttonConfiguration(
        primary: primary, compatibleWith: button.traitCollection,
        highlighted: button.isHighlighted, enabled: button.isEnabled)
      configuration.title = title
      button.configuration = configuration
    }
    button.titleLabel?.adjustsFontForContentSizeCategory = true
  }

  static func buttonConfiguration(
    primary: Bool, compatibleWith traits: UITraitCollection? = nil,
    highlighted: Bool = false, enabled: Bool = true
  ) -> UIButton.Configuration {
    var configuration = UIButton.Configuration.filled()
    let fill =
      primary
      ? (!enabled ? border : highlighted ? color(0x2C2C30) : self.primary)
      : (!enabled ? color(0xF9F9FA) : highlighted ? border : color(0xEFEFF0))
    let foreground = primary ? background : (!enabled ? color(0xC5C5C9) : text)
    configuration.baseBackgroundColor = fill
    configuration.baseForegroundColor = foreground
    configuration.background.backgroundColorTransformer = UIConfigurationColorTransformer { _ in
      fill
    }
    configuration.titleTextAttributesTransformer = UIConfigurationTextAttributesTransformer {
      incoming in
      var outgoing = incoming
      outgoing.font = font("Geist-Medium", size: 14, style: .body, compatibleWith: traits)
      outgoing.foregroundColor = foreground
      return outgoing
    }
    configuration.cornerStyle = .capsule
    configuration.titleAlignment = .center
    configuration.titleLineBreakMode = .byWordWrapping
    configuration.contentInsets = NSDirectionalEdgeInsets(
      top: 16, leading: 16, bottom: 16, trailing: 16)
    return configuration
  }

  private static func color(_ value: UInt32) -> UIColor {
    UIColor(
      red: CGFloat((value >> 16) & 0xFF) / 255,
      green: CGFloat((value >> 8) & 0xFF) / 255,
      blue: CGFloat(value & 0xFF) / 255, alpha: 1)
  }
}
