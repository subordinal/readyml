# Contributing

The most useful contribution is a startup failure we do not know about yet.

## Add a known issue

Records live in `src/readyml/data/known_issues.json`. One record is one upstream issue. Copy an existing record and fill in what the issue actually says:

- `architectures` or `models`: at least one, or the record can never match.
- `quantization`, `gpu_families`, `flags`: only when they are part of the trigger.
- `versions.affected` and `versions.fixed_in`: only what the issue or its fix states. Leave `null` when unknown.
- `symptom`: the key error line, quoted.
- `cause` and `workaround`: one plain sentence each.

Do not guess. A record with an empty field is better than a record with a wrong one. If you have no upstream issue, open a "Report a startup failure" issue here instead.

## Correct a record

Open a pull request that changes the record and say what in the upstream issue supports the change.

## Code

```
PYTHONPATH=src python -m unittest discover -s tests
```

Standard library only: no runtime dependencies. Add a test with any change to a check. Tests must not touch the network; use the fixtures in `tests/fixtures`.

## Rebuild the release indexes

```
PYTHONPATH=src python tools/build_support.py
python tools/build_sglang_support.py /path/to/sglang
```
