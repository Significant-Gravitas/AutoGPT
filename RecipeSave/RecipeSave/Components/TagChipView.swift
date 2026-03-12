import SwiftUI

struct TagChipView: View {
    let tag: String
    var isSelected: Bool = false
    var onRemove: (() -> Void)? = nil
    var onTap: (() -> Void)? = nil

    var body: some View {
        Button(action: { onTap?() }) {
            HStack(spacing: 4) {
                Text(tag)
                    .font(.caption)
                    .fontWeight(.medium)

                if let onRemove {
                    Button(action: onRemove) {
                        Image(systemName: "xmark")
                            .font(.system(size: 10, weight: .bold))
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(isSelected ? Color.accentColor : Color(.secondarySystemBackground))
            .foregroundStyle(isSelected ? .white : .primary)
            .clipShape(Capsule())
            .overlay(
                Capsule()
                    .strokeBorder(isSelected ? Color.clear : Color(.separator), lineWidth: 0.5)
            )
        }
        .buttonStyle(.plain)
    }
}
