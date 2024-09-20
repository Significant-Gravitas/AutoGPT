class Info {
  final String difficulty;
  final String description;
  final List<String> sideEffects;

  Info({
    required this.difficulty,
    required this.description,
    required this.sideEffects,
  });

  factory Info.fromJson(Map<String, dynamic> json) {
    return Info(
      difficulty: json['difficulty'] ?? "",
      description: json['description'] ?? "",
      sideEffects: List<String>.from(json['side_effects'] ?? []),
    );
  }
}
