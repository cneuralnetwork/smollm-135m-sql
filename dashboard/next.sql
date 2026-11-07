-- Run exactly one more SQL inference step, then show the full decoded output.
SELECT 'shell' AS component, 'Generating · SmolLM SQL Studio' AS title;
SELECT 'hero' AS component,
       'Generating one more token' AS title,
       'SQLite is running all 30 transformer layers. The decoded result will appear here.' AS description;

INSERT INTO generation_step(step_index)
SELECT (SELECT coalesce(max(step_index), -1) + 1 FROM generation_step)
WHERE (SELECT next_token_id FROM generated ORDER BY pos DESC LIMIT 1) <> 0;

SELECT 'code' AS component;
SELECT 'Generated text' AS title, text AS contents
FROM generated_text;

SELECT 'button' AS component;
SELECT 'Generate one more token' AS title,
       'next.sql' AS link,
       'plus' AS icon,
       'primary' AS color
WHERE (SELECT next_token_id FROM generated ORDER BY pos DESC LIMIT 1) <> 0;
SELECT 'Back to studio' AS title, 'index.sql' AS link, 'arrow-left' AS icon;
