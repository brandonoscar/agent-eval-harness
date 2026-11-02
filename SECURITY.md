# Security policy

## Reporting a vulnerability

Report it privately on GitHub: open the **Security** tab and choose **Report a vulnerability**. Please don't open a public issue for it.

Say what you ran, what happened and what should have happened. Expect a first reply within a week. Once a fix ships, the advisory is published and credits you, unless you'd rather not be named.

## What counts

- A safety assertion that passes when the reply or the tool calls break it (a leaked value, a forbidden tool call).
- A safety failure that gets re-rolled, averaged away, or reported as quality or infrastructure.
- An LLM judge that gates without passing its validation bar.
- A case file that makes the runner do anything beyond reading it.

## Supported versions

Only the latest release gets fixes.
