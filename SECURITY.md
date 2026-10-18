# Security Policy

## What this repository contains

This repository contains source code, configuration templates, documentation and
tests only. It does **not** contain model weights, datasets, trained LoRA
adapters, credentials, API tokens, SSH keys, personal filesystem paths,
hostnames, or IP addresses. Model weights and user datasets are downloaded or
created locally and are excluded by `.gitignore`.

## Reporting a vulnerability

If you find a security vulnerability, or believe private data (a credential, a
personal path, a dataset sample, or trained weights) has accidentally been
committed, please report it privately through GitHub's **Security → Report a
vulnerability** flow for this repository. Do not open a public issue that
repeats the sensitive value, and do not include live credentials in a report.

Please include the affected file and a short description of the impact. Reports
about accidentally committed private data are treated with the same priority as
code vulnerabilities: the value will be removed from the working tree, and the
repository history rewritten if it was ever committed.
