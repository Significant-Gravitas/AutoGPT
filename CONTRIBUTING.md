# AutoGPT Contribution Guide
Thanks for helping with AutoGPT! This page is the short version: what we ask of every
pull request, and where the detailed guides are.

[dev channel]: https://discord.com/channels/1092243196446249134/1095817829405704305

## Contributing to the AutoGPT Platform Folder
All contributions to [the autogpt_platform folder](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpt_platform) will be under our [Contribution License Agreement](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpt_platform/Contributor%20License%20Agreement%20(CLA).md). By making a pull request contributing to this folder, you agree to the terms of our CLA for your contribution. All contributions to other folders will be under the MIT license.

## In short
1. Avoid duplicate work, issues, PRs etc.
2. We encourage you to collaborate with fellow community members on bigger changes.
   * We highly recommend to post your idea and discuss it in the [dev channel].
3. Create a draft PR when starting work on bigger changes.
4. Open your PR against the `dev` branch and follow the conventions in [AGENTS.md](AGENTS.md).
5. Clearly explain your changes when submitting a PR.
6. Don't submit broken code: test/validate your changes.
7. Avoid making unnecessary changes, especially if they're purely based on your personal
   preferences. Doing so is the maintainers' job. ;-)
8. Please also consider contributing something other than code: improving the docs,
   reporting bugs, reviewing PRs and helping others on Discord all count.

## Pull requests
- Branch from `dev` and open your PR against `dev`. `master` is the release branch.
- Give the PR a [Conventional Commits](https://www.conventionalcommits.org/) title with a
  scope, such as `fix(backend): handle empty block input`.
- Fill in the [pull request template](.github/PULL_REQUEST_TEMPLATE.md).
- Run the linters and tests for the part you changed before you push.

## Guides
| To | Read |
|----|------|
| Run the platform locally | [Setting Up AutoGPT (Self-Host)](docs/platform/getting-started.md) |
| Follow the code style and conventions | [AGENTS.md](AGENTS.md) |
| Work on the frontend | [Frontend contributing guide](autogpt_platform/frontend/CONTRIBUTING.md) and [frontend testing](autogpt_platform/frontend/TESTING.md) |
| Test the backend | [Backend testing guide](autogpt_platform/backend/TESTING.md) |
| Build a block | [Build your own Blocks](docs/platform/new_blocks.md) and the [Block SDK Guide](docs/platform/block-sdk-guide.md) |
| Change the documentation | [Contributing to the Docs](docs/platform/contributing/contributing-to-the-docs.md) |
| Report a security issue | [Security policy](SECURITY.md) |

Everyone taking part is expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

If you wish to involve with the project (beyond just contributing PRs), please read the
wiki page about [Catalyzing](https://github.com/Significant-Gravitas/AutoGPT/wiki/Catalyzing).

Hop on our Discord. See you there! :-)

❤️ & 🔆
The team @ AutoGPT
https://discord.gg/autogpt
