import UIKit

@MainActor
final class ServerSettingsViewController: UIViewController, UITextFieldDelegate {
  private let address: String
  private let onConnect: (String) -> Void
  private let addressField = UITextField()
  private var isFinishing = false

  init(address: String, onConnect: @escaping (String) -> Void) {
    self.address = address
    self.onConnect = onConnect
    super.init(nibName: nil, bundle: nil)
    modalPresentationStyle = .pageSheet
    overrideUserInterfaceStyle = .light
    sheetPresentationController?.detents = [.large()]
    sheetPresentationController?.prefersGrabberVisible = true
  }

  required init?(coder: NSCoder) { nil }

  override func viewDidLoad() {
    super.viewDidLoad()
    view.backgroundColor = NativeTheme.background
    view.tintColor = NativeTheme.primary

    let heading = UILabel()
    heading.text = "Server settings"
    heading.font = NativeTheme.font(
      "Poppins-Medium", size: 22, style: .title2, compatibleWith: traitCollection)
    heading.textColor = NativeTheme.text
    heading.numberOfLines = 0
    heading.adjustsFontForContentSizeCategory = true
    heading.accessibilityTraits.insert(.header)
    heading.accessibilityIdentifier = "Server settings title"

    let message = UILabel()
    message.text =
      "Connect to AutoGPT or your own HTTPS server. Changing servers clears this app's saved website session."
    message.font = NativeTheme.font(
      "Geist-Regular", size: 14, style: .body, compatibleWith: traitCollection)
    message.textColor = NativeTheme.secondaryText
    message.numberOfLines = 0
    message.adjustsFontForContentSizeCategory = true

    let label = UILabel()
    label.text = "Server address"
    label.font = NativeTheme.font(
      "Geist-Medium", size: 14, style: .body, compatibleWith: traitCollection)
    label.textColor = NativeTheme.text
    label.numberOfLines = 0
    label.adjustsFontForContentSizeCategory = true

    addressField.text = address
    addressField.placeholder = "https://platform.agpt.co"
    addressField.accessibilityLabel = "Server address"
    addressField.font = NativeTheme.font(
      "Geist-Regular", size: 16, style: .body, compatibleWith: traitCollection)
    addressField.adjustsFontForContentSizeCategory = true
    addressField.textColor = NativeTheme.text
    addressField.backgroundColor = NativeTheme.background
    addressField.keyboardType = .URL
    addressField.textContentType = .URL
    addressField.autocapitalizationType = .none
    addressField.autocorrectionType = .no
    addressField.spellCheckingType = .no
    addressField.returnKeyType = .go
    addressField.layer.cornerRadius = 12
    addressField.layer.borderWidth = 1
    addressField.layer.borderColor = NativeTheme.border.cgColor
    addressField.leftView = UIView(frame: CGRect(x: 0, y: 0, width: 14, height: 1))
    addressField.leftViewMode = .always
    addressField.rightView = UIView(frame: CGRect(x: 0, y: 0, width: 14, height: 1))
    addressField.rightViewMode = .always
    addressField.delegate = self

    let connect = UIButton(type: .system)
    NativeTheme.styleButton(connect, primary: true)
    connect.setTitle("Connect", for: .normal)
    connect.addAction(UIAction { [weak self] _ in self?.connect() }, for: .touchUpInside)
    let cancel = UIButton(type: .system)
    NativeTheme.styleButton(cancel, primary: false)
    cancel.setTitle("Cancel", for: .normal)
    cancel.addAction(UIAction { [weak self] _ in self?.cancel() }, for: .touchUpInside)

    let stack = UIStackView(arrangedSubviews: [
      heading, message, label, addressField, connect, cancel,
    ])
    stack.axis = .vertical
    stack.spacing = 16
    stack.setCustomSpacing(24, after: message)
    stack.setCustomSpacing(8, after: label)
    stack.setCustomSpacing(24, after: addressField)
    stack.translatesAutoresizingMaskIntoConstraints = false
    let scroll = UIScrollView()
    scroll.accessibilityIdentifier = "Server settings content"
    scroll.keyboardDismissMode = .interactive
    scroll.translatesAutoresizingMaskIntoConstraints = false
    view.addSubview(scroll)
    scroll.addSubview(stack)
    let width = stack.widthAnchor.constraint(
      equalTo: scroll.frameLayoutGuide.widthAnchor, constant: -48)
    width.priority = .defaultHigh
    NSLayoutConstraint.activate([
      scroll.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
      scroll.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor),
      scroll.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor),
      scroll.bottomAnchor.constraint(equalTo: view.keyboardLayoutGuide.topAnchor),
      scroll.contentLayoutGuide.widthAnchor.constraint(
        equalTo: scroll.frameLayoutGuide.widthAnchor),
      stack.topAnchor.constraint(equalTo: scroll.contentLayoutGuide.topAnchor, constant: 24),
      stack.bottomAnchor.constraint(equalTo: scroll.contentLayoutGuide.bottomAnchor, constant: -24),
      stack.centerXAnchor.constraint(equalTo: scroll.frameLayoutGuide.centerXAnchor),
      stack.leadingAnchor.constraint(
        greaterThanOrEqualTo: scroll.contentLayoutGuide.leadingAnchor, constant: 24),
      stack.trailingAnchor.constraint(
        lessThanOrEqualTo: scroll.contentLayoutGuide.trailingAnchor, constant: -24),
      stack.widthAnchor.constraint(lessThanOrEqualToConstant: 416),
      width,
      addressField.heightAnchor.constraint(greaterThanOrEqualToConstant: 46),
      connect.heightAnchor.constraint(greaterThanOrEqualToConstant: 52),
      cancel.heightAnchor.constraint(greaterThanOrEqualToConstant: 52),
    ])
  }

  func textFieldShouldReturn(_ textField: UITextField) -> Bool {
    connect()
    return true
  }

  private func connect() {
    guard !isFinishing else { return }
    isFinishing = true
    let value = addressField.text ?? ""
    let completion = onConnect
    view.endEditing(true)
    dismiss(animated: true) { completion(value) }
  }

  private func cancel() {
    guard !isFinishing else { return }
    isFinishing = true
    view.endEditing(true)
    dismiss(animated: true)
  }
}
