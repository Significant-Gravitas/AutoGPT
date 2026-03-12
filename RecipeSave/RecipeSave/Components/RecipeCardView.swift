import SwiftUI

struct RecipeCardView: View {
    let recipe: Recipe

    var body: some View {
        HStack(spacing: 12) {
            thumbnailView
                .frame(width: 80, height: 80)
                .clipShape(RoundedRectangle(cornerRadius: 10))

            VStack(alignment: .leading, spacing: 6) {
                Text(recipe.title)
                    .font(.headline)
                    .lineLimit(2)

                if !recipe.ingredients.isEmpty {
                    Text("\(recipe.ingredients.count) ingredients")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

                if !recipe.tags.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 4) {
                            ForEach(recipe.tags.prefix(3), id: \.self) { tag in
                                TagChipView(tag: tag)
                            }
                        }
                    }
                }
            }

            Spacer(minLength: 0)
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
    }

    @ViewBuilder
    private var thumbnailView: some View {
        if let data = recipe.thumbnailData, let uiImage = UIImage(data: data) {
            Image(uiImage: uiImage)
                .resizable()
                .scaledToFill()
        } else {
            ZStack {
                Color(.tertiarySystemBackground)
                Image(systemName: "fork.knife")
                    .font(.system(size: 30))
                    .foregroundStyle(.tertiary)
            }
        }
    }
}
