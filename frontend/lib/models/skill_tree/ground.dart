class Ground {
  final String answer;
  final List<String> shouldContain;
  final List<String> shouldNotContain;
  final List<String> files;
  final Map<String, dynamic> eval;

  Ground({
    required this.answer,
    required this.shouldContain,
    required this.shouldNotContain,
    required this.files,
    required this.eval,
  });

  factory Ground.fromJson(Map<String, dynamic> json) {
    return Ground(
      answer: json['answer'] ?? "",
      shouldContain: List<String>.from(json['should_contain'] ?? []),
      shouldNotContain: List<String>.from(json['should_not_contain'] ?? []),
      files: List<String>.from(json['files'] ?? []),
      eval: json['eval'] ?? {},
    );
  }
}
