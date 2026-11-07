-- Run in a fresh model database after setup.sql (or after 15_reset.sql).
-- The sqlite3 CLI needs PRAGMA recursive_triggers on for each new connection.
PRAGMA recursive_triggers = ON;
PRAGMA cache_size = -100000;
PRAGMA temp_store = MEMORY;
INSERT INTO prompt_input(prompt_id, text) VALUES (1, 'Hello');
INSERT INTO prompt_run(prompt_id) VALUES (1);
SELECT next_token_id FROM generated ORDER BY pos DESC LIMIT 1;
SELECT text FROM generated_text;

-- To get three output tokens total, run this after the first prediction:
-- INSERT INTO generation_request(id, additional_steps) VALUES (1, 2);
-- SELECT text FROM generated_text;
