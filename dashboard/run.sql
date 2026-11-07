-- SQLPage executes this file for the prompt form. The model work is SQL.
SELECT 'shell' AS component,
       'Running · SmolLM / SQL' AS title,
       'SMOLLM / SQL' AS navbar_title,
       'database' AS icon,
       '/theme.css' AS css,
       'Local inference · Built with SQLPage' AS footer;
SELECT 'text' AS component,
       'Generating in SQL' AS title,
       'The model is working in SQLite. Decoded text will appear below, one token at a time.' AS contents;

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

INSERT INTO prompt_input(prompt_id, text) VALUES (1, :prompt);
-- SQLPage can reuse this connection, so reset its TEMP cursor every run.
CREATE TEMP TABLE IF NOT EXISTS dashboard_cursor (printed_bytes INTEGER NOT NULL);
DELETE FROM dashboard_cursor;
INSERT INTO dashboard_cursor VALUES (0);
CREATE TEMP VIEW IF NOT EXISTS dashboard_delta AS
SELECT CAST(substr(CAST(g.text AS BLOB), c.printed_bytes + 1) AS TEXT) AS text
FROM generated_text AS g CROSS JOIN dashboard_cursor AS c
WHERE length(CAST(g.text AS BLOB)) > c.printed_bytes;

SELECT 'html' AS component, '<section class="run-output">' AS html;
SELECT 'text' AS component, 'LIVE OUTPUT' AS title;
INSERT INTO prompt_run(prompt_id) VALUES (1);
SELECT text AS contents FROM dashboard_delta;
UPDATE dashboard_cursor SET printed_bytes =
  (SELECT length(CAST(text AS BLOB)) FROM generated_text);

INSERT INTO generation_step(step_index)
SELECT 0 WHERE coalesce(CAST(:tokens AS INTEGER), 2) >= 2
  AND (SELECT next_token_id FROM generated ORDER BY pos DESC LIMIT 1) <> 0;
SELECT text AS contents FROM dashboard_delta;
UPDATE dashboard_cursor SET printed_bytes =
  (SELECT length(CAST(text AS BLOB)) FROM generated_text);

INSERT INTO generation_step(step_index)
SELECT 1 WHERE coalesce(CAST(:tokens AS INTEGER), 2) >= 3
  AND (SELECT next_token_id FROM generated ORDER BY pos DESC LIMIT 1) <> 0;
SELECT text AS contents FROM dashboard_delta;
UPDATE dashboard_cursor SET printed_bytes =
  (SELECT length(CAST(text AS BLOB)) FROM generated_text);

INSERT INTO generation_step(step_index)
SELECT 2 WHERE coalesce(CAST(:tokens AS INTEGER), 2) >= 4
  AND (SELECT next_token_id FROM generated ORDER BY pos DESC LIMIT 1) <> 0;
SELECT text AS contents FROM dashboard_delta;
UPDATE dashboard_cursor SET printed_bytes =
  (SELECT length(CAST(text AS BLOB)) FROM generated_text);

INSERT INTO generation_step(step_index)
SELECT 3 WHERE coalesce(CAST(:tokens AS INTEGER), 2) >= 5
  AND (SELECT next_token_id FROM generated ORDER BY pos DESC LIMIT 1) <> 0;
SELECT text AS contents FROM dashboard_delta;
UPDATE dashboard_cursor SET printed_bytes =
  (SELECT length(CAST(text AS BLOB)) FROM generated_text);

INSERT INTO generation_step(step_index)
SELECT 4 WHERE coalesce(CAST(:tokens AS INTEGER), 2) >= 6
  AND (SELECT next_token_id FROM generated ORDER BY pos DESC LIMIT 1) <> 0;
SELECT text AS contents FROM dashboard_delta;
UPDATE dashboard_cursor SET printed_bytes =
  (SELECT length(CAST(text AS BLOB)) FROM generated_text);

INSERT INTO generation_step(step_index)
SELECT 5 WHERE coalesce(CAST(:tokens AS INTEGER), 2) >= 7
  AND (SELECT next_token_id FROM generated ORDER BY pos DESC LIMIT 1) <> 0;
SELECT text AS contents FROM dashboard_delta;
UPDATE dashboard_cursor SET printed_bytes =
  (SELECT length(CAST(text AS BLOB)) FROM generated_text);

INSERT INTO generation_step(step_index)
SELECT 6 WHERE coalesce(CAST(:tokens AS INTEGER), 2) >= 8
  AND (SELECT next_token_id FROM generated ORDER BY pos DESC LIMIT 1) <> 0;
SELECT text AS contents FROM dashboard_delta;
UPDATE dashboard_cursor SET printed_bytes =
  (SELECT length(CAST(text AS BLOB)) FROM generated_text);

SELECT 'code' AS component, 'result-console' AS class;
SELECT 'Generated text' AS title, text AS contents
FROM generated_text;

SELECT 'button' AS component, 'session-actions' AS class;
SELECT 'Generate one more token' AS title,
       'next.sql' AS link,
       'plus' AS icon,
       'primary' AS color
WHERE (SELECT next_token_id FROM generated ORDER BY pos DESC LIMIT 1) <> 0;
SELECT 'Back to studio' AS title, 'index.sql' AS link, 'arrow-left' AS icon;
SELECT 'html' AS component, '</section>' AS html;
