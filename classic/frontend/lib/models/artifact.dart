/// `Artifact` class represents an artifact either created by or submitted to the agent.
///
/// Each artifact object contains an ID, a flag indicating if it was created by the agent,
/// a file name, and a relative path of the artifact in the agent's workspace.
class Artifact {
  // ID of the artifact.
  final String artifactId;

  // Whether the artifact has been created by the agent.
  final bool agentCreated;

  // Filename of the artifact.
  final String fileName;

  // Relative path of the artifact in the agent's workspace.
  final String? relativePath;

  /// Creates an `Artifact` instance.
  ///
  /// - `artifactId`: ID of the artifact. This is a required field.
  /// - `agentCreated`: Indicates whether the artifact was created by the agent. This is a required field.
  /// - `fileName`: The file name of the artifact. This is a required field.
  /// - `relativePath`: The relative path of the artifact in the agent's workspace. This field can be null.
  Artifact({
    required this.artifactId,
    required this.agentCreated,
    required this.fileName,
    this.relativePath,
  });

  /// Creates an `Artifact` instance from a map.
  ///
  /// This constructor is used for deserializing a JSON object into an `Artifact` instance.
  /// It expects all the required fields to be present; otherwise, an error will be thrown.
  ///
  /// - `map`: The map from which the `Artifact` instance will be created.
  factory Artifact.fromJson(Map<String, dynamic> map) {
    if (map['artifact_id'] == null ||
        map['agent_created'] == null ||
        map['file_name'] == null) {
      throw const FormatException(
          'Invalid JSON: Missing one of the required fields.');
    }

    return Artifact(
      artifactId: map['artifact_id'],
      agentCreated: map['agent_created'],
      fileName: map['file_name'],
      relativePath: map['relative_path'],
    );
  }

  /// Converts the `Artifact` instance into a JSON object.
  ///
  /// This can be useful for encoding the `Artifact` object into a JSON string.
  Map<String, dynamic> toJson() => {
        'artifact_id': artifactId,
        'agent_created': agentCreated,
        'file_name': fileName,
        'relative_path': relativePath,
      };
}
