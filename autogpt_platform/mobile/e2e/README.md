# Native journeys

Use an installed simulator build from `scripts/build-simulator.sh` or an ordinary Release build, and Maestro 2.11 with Java 21. Open the simulator in the shared T3 Device panel before local testing. The launch journey needs no account or server:

```sh
maestro --device <simulator-udid> test --format junit --output e2e/artifacts/launch.xml e2e/launch.yaml
python3 scripts/check-junit.py e2e/artifacts/launch.xml
```

`launch.yaml` verifies the native server selector, a clean first launch without storage errors, and rejection of nonlocal plain HTTP addresses. The macOS CI lane runs it against the optimized SwiftUI Release build, then installs Debug and runs `apple/e2e/ui-protocol-fixtures.yaml`. The latter covers a real Keychain round trip and deterministic stream/card/editor/artifact interactions; Debug-only fixtures are excluded from Release. Neither journey certifies the server chat integration below.

## Production-server chat journey

Use a dedicated local test stack with the parent PR's production Better Auth handler, REST API, DatabaseManager, copilot executor, migrated PostgreSQL database, three-node Redis cluster and RabbitMQ. Enable email signup without verification only in that isolated test configuration. Never point this harness at a production account or shared application database. It creates a fresh `native-…@example.test` account for each run and retains its conversation for inspection; dispose of the isolated test database after the experiment.

The only scripted service is the external model. Run it with the existing backend Python environment, from the repository root:

```sh
poetry -C autogpt_platform/backend run python ../mobile/e2e/scripted_model.py --port 18810
```

Configure the test copilot executor with `CHAT_USE_LOCAL=true`, `CHAT_BASE_URL=http://127.0.0.1:18810/v1`, `CHAT_FAST_STANDARD_MODEL=native-e2e`, `CHAT_THINKING_STANDARD_MODEL=native-e2e`, `CHAT_USE_OPENROUTER=false`, `CHAT_USE_E2B_SANDBOX=false` and `CHAT_TEST_MODE=false`. Give the local provider client a generated nonempty `CHAT_API_KEY`; the scripted boundary does not authenticate it. Use local workspace storage. The native app, auth handlers, authorization, queues, tools, stream, transcript persistence and history API remain production code.

From this mobile folder, run:

```sh
python3 scripts/run-chat-journey.py \
  --device <simulator-udid> \
  --auth-origin http://127.0.0.1:18800 \
  --api-origin http://127.0.0.1:18806
```

The two origins must match the isolated stack; those ports are examples. `--maestro /absolute/path/to/maestro` can select a local tool installation. The helper generates the test password privately and passes it through environment variables. It drives email sign-in, dismisses the system password-save sheet, opens Otto, sends a message, checks the response, terminates and relaunches the app, and checks the restored transcript. It then verifies through the production API that exactly one user message and one assistant response were persisted.

Reports and images go to a private directory under ignored `e2e/artifacts/`. The helper redacts the generated password from its text reports. Inspect screenshots before sharing them; publish conversation images, not sign-in or password-manager screens.

This journey currently covers the baseline chat engine against a scripted model, not a real model provider or the SDK engine. Google, expert kickoff, card actions, disconnect/retry, session revocation, account/server switching and other requested features need additional journeys as they are implemented. The complete local service setup is not automated on the macOS CI runner yet; cold/warm launch-gate measurements are separate evidence.
