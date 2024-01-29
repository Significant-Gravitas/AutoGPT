// TODO: Remove the ability to have null values when benchmark implementation is complete
/// `Config` holds configuration settings related to the benchmark run.
///
/// It contains the path to the benchmark configuration for the agent and
/// the host address where the benchmark is running.
class Config {
  /// The path to the configuration file for the agent's benchmark.
  /// This is typically a JSON file specifying various settings and parameters
  /// for the benchmark run.
  final String agentBenchmarkConfigPath;

  /// The host address where the benchmark is running.
  /// This could be a local or remote server address.
  final String host;

  /// Constructs a new `Config` instance.
  ///
  /// [agentBenchmarkConfigPath]: The path to the agent's benchmark configuration file.
  /// [host]: The host address where the benchmark is running.
  Config({
    required this.agentBenchmarkConfigPath,
    required this.host,
  });

  /// Creates a `Config` instance from a map.
  ///
  /// [json]: A map containing key-value pairs corresponding to `Config` fields.
  ///
  /// Returns a new `Config` populated with values from the map.
  factory Config.fromJson(Map<String, dynamic> json) => Config(
        agentBenchmarkConfigPath:
            json['agent_benchmark_config_path'] ?? 'placeholder',
        host: json['host'] ?? 'https://github.com/Significant-Gravitas/AutoGPT',
      );

  /// Converts the `Config` instance to a map.
  ///
  /// Returns a map containing key-value pairs corresponding to `Config` fields.
  Map<String, dynamic> toJson() => {
        'agent_benchmark_config_path': agentBenchmarkConfigPath,
        'host': host,
      };
}
