-- Clear one inference session; keep imported model and tokenizer data.
BEGIN;
DELETE FROM generation_request;
DELETE FROM generation_step;
DELETE FROM prompt_run;
DELETE FROM prompt_input;
DELETE FROM bpe_choice;
DELETE FROM bpe_step;
DELETE FROM bpe_symbol;
DELETE FROM bpe_piece;
DELETE FROM decode_token;
DELETE FROM generated;
DELETE FROM logit_chunk;
DELETE FROM logit_score;
DELETE FROM final_job;
DELETE FROM layer_done;
DELETE FROM layer_job;
DELETE FROM attention_context;
DELETE FROM attention_prob;
DELETE FROM attention_score;
DELETE FROM key_cache;
DELETE FROM value_cache;
DELETE FROM q_rot;
DELETE FROM k_rot;
DELETE FROM q_raw;
DELETE FROM k_raw;
DELETE FROM v_raw;
DELETE FROM gate_vector;
DELETE FROM up_vector;
DELETE FROM norm_vector;
DELETE FROM hidden;
DELETE FROM matrix_job;
DELETE FROM matrix_input;
DELETE FROM matrix_output;
DELETE FROM scalar_request;
DELETE FROM inference_input;
UPDATE generation_config SET prompt_length = 1 WHERE id = 1;
UPDATE inference_settings SET last_layer = 29, emit_logits = 1 WHERE id = 1;
COMMIT;

-- Run from the repository root with:
--   sqlite3 outputs/smollm.db < outputs/stream.sql
-- Each SELECT returns only newly decoded text. The CLI prints a chunk as
-- soon as the corresponding inference statement completes.
-- Edit the prompt below. Eight output tokens are requested by default.
PRAGMA recursive_triggers = ON;
PRAGMA cache_size = -100000;
PRAGMA temp_store = MEMORY;

CREATE TEMP TABLE stream_cursor (printed_bytes INTEGER NOT NULL);
INSERT INTO stream_cursor VALUES (0);
CREATE TEMP VIEW stream_delta AS
SELECT CAST(substr(CAST(g.text AS BLOB), c.printed_bytes + 1) AS TEXT) AS text
FROM generated_text AS g CROSS JOIN stream_cursor AS c
WHERE length(CAST(g.text AS BLOB)) > c.printed_bytes;

INSERT INTO prompt_input(prompt_id, text) VALUES (1, 'Hello');
INSERT INTO prompt_run(prompt_id) VALUES (1);
SELECT text FROM stream_delta;
UPDATE stream_cursor SET printed_bytes =
  (SELECT length(CAST(text AS BLOB)) FROM generated_text);

INSERT INTO generation_step(step_index) VALUES (0);
SELECT text FROM stream_delta;
UPDATE stream_cursor SET printed_bytes =
  (SELECT length(CAST(text AS BLOB)) FROM generated_text);

INSERT INTO generation_step(step_index) VALUES (1);
SELECT text FROM stream_delta;
UPDATE stream_cursor SET printed_bytes =
  (SELECT length(CAST(text AS BLOB)) FROM generated_text);

INSERT INTO generation_step(step_index) VALUES (2);
SELECT text FROM stream_delta;
UPDATE stream_cursor SET printed_bytes =
  (SELECT length(CAST(text AS BLOB)) FROM generated_text);

INSERT INTO generation_step(step_index) VALUES (3);
SELECT text FROM stream_delta;
UPDATE stream_cursor SET printed_bytes =
  (SELECT length(CAST(text AS BLOB)) FROM generated_text);

INSERT INTO generation_step(step_index) VALUES (4);
SELECT text FROM stream_delta;
UPDATE stream_cursor SET printed_bytes =
  (SELECT length(CAST(text AS BLOB)) FROM generated_text);

INSERT INTO generation_step(step_index) VALUES (5);
SELECT text FROM stream_delta;
UPDATE stream_cursor SET printed_bytes =
  (SELECT length(CAST(text AS BLOB)) FROM generated_text);

INSERT INTO generation_step(step_index) VALUES (6);
SELECT text FROM stream_delta;
UPDATE stream_cursor SET printed_bytes =
  (SELECT length(CAST(text AS BLOB)) FROM generated_text);
