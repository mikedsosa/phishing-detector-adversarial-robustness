# Security policy

This repository is a **lab project, not production software**. It is a research project, not a production mail or browser filter. Do not rely on these models to block phishing.

## Reporting a problem

If you find a security issue (for example a query, script or config that could cause harm if copied as is, or anything that looks like an exposed secret), please report it privately through GitHub's **[Report a vulnerability](https://github.com/mikedsosa/phishing-detector-adversarial-robustness/security/advisories/new)** form rather than opening a public issue. Ordinary bugs and corrections can go in a normal issue.

Please include the file and line, what you expected and what happened. Reports are handled on a best-effort basis; there is no SLA.

## Secrets and data

- Never include credentials, tokens, tenant IDs or real user data in an issue, pull request or report.
- This repo is meant to contain no secrets. Commits are scanned with `gitleaks`.
- The attacks rewrite URL strings offline; no URL is ever resolved or visited. Please keep it that way in contributions.

## Supported versions

Only the latest commit on `main` is maintained.
