# Security Policy

## Scope

`cleanroom-dsp-profiler` is a clean-room audio system-identification toolkit and zero-allocation Pure-Rust DSP starter library. Security-relevant issues include:

- Memory safety issues or buffer overruns in the Rust C ABI FFI layer (`dsp-core-rust/src/lib.rs`)
- Unsafe deserialization or path traversal in CLI / config / WAV / `.nam` export utilities
- Accidental credential or PII exposure in CI workflows or templates

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, use GitHub's **Private Vulnerability Reporting** feature under the repository's **Security -> Advisories -> Report a vulnerability** tab.

Please include:
- A clear description of the vulnerability and impact
- Minimal steps to reproduce (or a failing unit test)
- Affected commit SHA or release tag

We will acknowledge reports promptly and coordinate a fix and disclosure timeline through GitHub Security Advisories.
