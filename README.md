# SmolLM-135M, entirely in SQL

> A real 135M parameter language model running inside SQLite. Tokenization, transformer math, attention, KV caching, and greedy text generation are SQL statements, views, and triggers. Its CLI example streams decoded text as it runs.

This project loads the official [HuggingFaceTB/SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M) weights directly from `safetensors` into SQLite. The inference engine uses no Python, JavaScript, extensions, or host language functions. The `sqlite3` command line program is just the SQL runner.

```text
prompt text
    │
    ▼
Unicode pretokenizer → byte-level BPE → token IDs
    │
    ▼
embedding → 30 transformer layers → final norm → 49,152 logits
                 │                               │
                 └────── SQL KV cache             ▼
                                            greedy next token
                                                   │
                                                   ▼
                                             decoded text
```

## Try it

Run all commands from the repository root. You need the `sqlite3` CLI with `readfile()`, JSON, math functions, and `unhex()`; this was tested with SQLite 3.51. You also need about 6 GB of free disk space and considerable patience: the one-time weight conversion is lengthy, and generation takes roughly a minute or more per token on the tested machine.

```sh
git clone https://github.com/cneuralnetwork/smollm-135m-sql.git
cd smollm-135m-sql

# Get the upstream data (kept outside Git).
curl -L --fail --output outputs/model.safetensors \
  https://huggingface.co/HuggingFaceTB/SmolLM-135M/resolve/main/model.safetensors
curl -L --fail --output outputs/tokenizer.json \
  https://huggingface.co/HuggingFaceTB/SmolLM-135M/resolve/main/tokenizer.json
curl -L --fail --output outputs/config.json \
  https://huggingface.co/HuggingFaceTB/SmolLM-135M/resolve/main/config.json
curl -L --fail --output outputs/UnicodeData.txt \
  https://www.unicode.org/Public/UCD/latest/ucd/UnicodeData.txt

# Import the model and tokenizer with SQL.
sqlite3 outputs/smollm.db < outputs/setup.sql

# Convert F32 weight bytes into indexed SQLite REAL values once.
# This is slow but resumable: rerun the command if interrupted.
sqlite3 outputs/smollm.db < outputs/accelerate.sql

# Stream eight decoded output chunks from "Hello" (one chunk per line).
sqlite3 outputs/smollm.db < outputs/stream.sql
```

The stream begins like this, with new text appearing after each inference step:

```text
,
 I
```

The stream prints text chunks, never numeric token IDs. Each chunk appears on its own line; a chunk can be a word, punctuation, or a word fragment because the model uses subword tokens. Edit the prompt in `outputs/stream.sql` to try another input, or remove or repeat its generation blocks to change the output length. The script resets inference state at the start, so you can rerun it directly. If you received this project as a local Codex deliverable, `outputs/smollm.db` may already be built. In a GitHub clone, build the database as shown above.

### Run the browser dashboard

The dashboard uses [SQLPage](https://sql-page.com/) to turn SQL files into a local web app. The authored dashboard and the inference engine are SQL; SQLPage supplies the web server and browser controls. Build `outputs/smollm.db` first, then from the repository root run:

```sh
brew install sqlpage                 # macOS; other systems: SQLPage releases
sqlpage --config-dir dashboard_config
```

If you have the local Codex deliverable, a verified SQLPage binary is also included outside Git:

```sh
./.local/bin/sqlpage --config-dir dashboard_config
```

Open **[http://127.0.0.1:8080](http://127.0.0.1:8080)**. Enter a prompt, choose 1–8 output tokens, and click **Run model**. Decoded text appears as each token finishes. You can then generate one more token or reset the session. Stop the server with `Ctrl+C`. Run only one inference session against the database at a time; stop a terminal stream before using the dashboard.

Run SQLPage from the repository root: the database path in `dashboard_config/sqlpage.json` is relative to that directory. If a page reports `no such table: generation_request`, check that `outputs/smollm.db` is the initialized model database and rerun `outputs/setup.sql` if needed. Do not use the dashboard against an empty SQLite file.

The server listens only on `127.0.0.1`, and its web root is the `dashboard/` folder, separate from the model database and configuration.

### Generate from your own prompt

Reset the inference state between prompts, then open the SQLite shell:

```sh
sqlite3 outputs/smollm.db < outputs/15_reset.sql
sqlite3 outputs/smollm.db
```

At the `sqlite>` prompt, run:

```sql
PRAGMA recursive_triggers = ON;
INSERT INTO prompt_input(prompt_id, text) VALUES (1, 'Once upon a time');
INSERT INTO prompt_run(prompt_id) VALUES (1);
SELECT text FROM generated_text;

-- Two more predictions, for three output tokens total.
INSERT INTO generation_request(id, additional_steps) VALUES (1, 2);
SELECT text FROM generated_text;
```

`prompt_run` processes every prompt token and produces the first prediction. `additional_steps` is the number of *further* predictions, so request `N - 1` for `N` output tokens. A new prompt needs another reset. End the shell with `.quit`.

## How it works

| Stage | SQL implementation |
| --- | --- |
| Model import | `readfile()` reads `model.safetensors`; SQL parses its header and slices all 272 F32 tensors. |
| Tokenization | Unicode categories, GPT-2 style pretokenization, byte mapping, and BPE merges run in SQL. |
| Dense layers | Relational matrix jobs and indexed `REAL` weights compute the projections. |
| Attention | SQL computes rotary positions, causal scores, softmax, and a persistent key/value cache. |
| Sampling | The final projection scores 49,152 vocabulary entries; SQL selects the maximum logit. |
| Generation | Triggers advance one token at a time and decode the resulting IDs. |

`outputs/setup.sql` is the combined, runnable setup. The numbered SQL files hold the same components separately for inspection and experiments. `outputs/accelerate.sql` materializes numeric weights for practical inference; it does not change the model.

### Speed

The accelerated matrix multiply scans one weight row at a time. This avoids a temporary `GROUP BY` tree for every projection. On one local test of the `Hello` first-token example, it reduced runtime from about 90 seconds to 44 seconds while producing the same output. Results depend on the machine and disk cache.

Fresh databases get this trigger from `accelerate.sql`. To upgrade a database built from an earlier version of this repository, run the following **after any active inference finishes**:

```sh
sqlite3 outputs/smollm.db < outputs/19_rowwise_matmul.sql
```

The next major speed target is storage layout: SQLite still visits roughly 135 million individual numeric weight rows per generated token. Compact row blocks or quantized weights could reduce that cost, though they need careful accuracy checks.

## Files

```text
README.md
dashboard/             SQLPage pages: prompt, live output, next token, reset
dashboard_config/      local-only server settings and SQLite connection setup
outputs/
  setup.sql           schema, import, tokenizer, and inference engine
  accelerate.sql      one-time numeric weight materialization
  stream.sql          incremental decoded text, eight output tokens
  example.sql         first-token text example
  15_reset.sql        clear prompt, generation, and KV state
  19_rowwise_matmul.sql upgrade an existing accelerated database
  00_*.sql … 18_*.sql component SQL sources
  config.json         download separately; ignored by Git
  tokenizer.json      download separately; ignored by Git
  UnicodeData.txt     download separately; ignored by Git
  model.safetensors   download separately; ignored by Git
  smollm.db           generated locally; ignored by Git
```

## Checks and limits

The full engine was exercised with a two-token run from `Hello`, producing `, I`. That run executed all 30 transformer layers and populated the KV cache. Import checks found all 272 expected tensors and 134,515,008 numeric weights; `PRAGMA quick_check` returned `ok`. The tokenizer was checked with Unicode and special tokens.

Generation is greedy and supports at most the model's 2,048-token context. This is a proof of concept for SQL inference, not a fast serving engine. The database is about 4.2 GB after acceleration. The upstream model weights are about 528 MB; they and the generated database stay outside Git.

To validate a fresh import before acceleration:

```sh
sqlite3 outputs/smollm.db 'SELECT count(*) FROM model_problem;'
sqlite3 outputs/smollm.db 'PRAGMA quick_check;'
```

The expected results are `0` and `ok`. You can check the downloaded weight file against its SHA-256 digest:

```sh
shasum -a 256 outputs/model.safetensors
# c7a387d6fe81ca6dd304aeb809bda3932ff1bbef3ca41c9484502f2f448dc093
```

Model and tokenizer: [HuggingFaceTB/SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M) (Apache 2.0). Unicode data: [Unicode Character Database](https://www.unicode.org/Public/UCD/latest/ucd/UnicodeData.txt) ([Unicode License](https://www.unicode.org/license.txt)).
