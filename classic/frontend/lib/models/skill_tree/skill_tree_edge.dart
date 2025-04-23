class SkillTreeEdge {
  final String id;
  final String from;
  final String to;
  final String arrows;

  SkillTreeEdge({
    required this.id,
    required this.from,
    required this.to,
    required this.arrows,
  });

  // Optionally, add a factory constructor to initialize from JSON
  factory SkillTreeEdge.fromJson(Map<String, dynamic> json) {
    return SkillTreeEdge(
      id: json['id'],
      from: json['from'],
      to: json['to'],
      arrows: json['arrows'],
    );
  }
}
