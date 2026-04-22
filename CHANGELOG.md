# Changelog

All notable changes to this project are documented in this file.

This project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Direct public hygiene audit script:
  - `.github/scripts/audit_public_hygiene.sh`
- GitHub Actions unit-test workflow for the public bridge test suite.
- GitHub issue and pull request templates.
- Public maintainer docs for contributors and users:
  - `CONTRIBUTING.md`
  - `SUPPORT.md`
  - `SECURITY.md`

### Changed

- `README.md` now reflects the files tracked on current `main`, removes
  mirror-era/bootstrap references, and documents the public privacy boundary.
- `CONTRIBUTING.md` now points at the direct public validation commands.
- `bridge/benchmark_midi_write.py` now reports missing composition-runtime
  modules clearly instead of failing during import.
- `.gitignore` now excludes runtime logs, local conversation memory, environment
  variants, and generated media artifacts.
