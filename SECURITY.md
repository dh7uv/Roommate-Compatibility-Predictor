# Security and privacy

This repository is an academic prototype and is not intended to be deployed with real student data without further review.

## Reporting a concern

Please do not publish private student information or exploit details in a public issue. Contact the repository owner privately through a channel listed on their GitHub profile.

## Maintainer checklist

- Keep real survey data out of public Git history.
- Rotate any credentials that may have been committed.
- Never run Flask with debug mode enabled on a public host.
- Validate and constrain all form inputs.
- Add appropriate authentication and authorization before exposing student profiles.
- Review how profile changes are persisted and who can read them.
