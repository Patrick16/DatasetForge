# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Git workflow

- **Never commit automatically.** Finish and verify the work (tests passing,
  live-tested where relevant), then stop and report back — wait for an
  explicit go-ahead from the user before running `git commit`. This applies
  even when a task clearly reads as "done" from your side.
- Same for `git push`: don't push without being asked, even to a feature
  branch.
- Pull requests are only opened in response to an explicit request — never
  inferred from a feature being complete.
