# Global instructions

## Language & authorship
- Talk to me in French in the chat (use "tu").
- Everything written to the repo or pushed (code comments, docs, README, commits,
  branch names, PRs, issues) must be in English.
- Never write my first name or personal details anywhere. Author/maintainer: "monolithoid".

## Environment
- Primary OS: Ubuntu (bash). Give Linux/bash commands by default.
- Secondary: Windows 11 (PowerShell) — only when I say I'm on Windows.
- Python: use a virtual environment; never install packages globally.

## Deployment
- Preferred target: my VPS managed by Coolify.
- Favor solutions packaged as Docker images published on Docker Hub.
- Provide a Dockerfile (and docker-compose.yml when several services are needed)
  that Coolify can deploy as-is.
- Prefer configuration through environment variables; never hardcode secrets.
- Prefer official, well-maintained base images; pin versions instead of "latest".

## Git
- Conventional Commits (feat:, fix:, docs:, chore:, refactor:).
- Never push, force-push or commit to main without asking me first.
- Never commit secrets (.env, API keys, tokens); check .gitignore.

## Working style
- Keep changes minimal and focused; don't refactor unrelated code.
- Ask before adding a new dependency.
- Run tests/build before saying something is done.
- When unsure about intent, ask instead of guessing.