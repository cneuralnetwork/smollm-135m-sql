-- Run exactly one more SQL inference step, then show the full decoded output.
SELECT 'shell' AS component,
       'Generating · SmolLM / SQL' AS title,
       'SMOLLM / SQL' AS navbar_title,
       'database' AS icon,
       '/theme.css' AS css,
       'Local inference · Built with SQLPage' AS footer;
SELECT 'text' AS component,
       'One more token' AS title,
       'SQLite is running the next forward pass. The decoded result will appear below.' AS contents;

SELECT 'html' AS component, '<section class="run-output">' AS html;
INSERT INTO generation_step(step_index)
SELECT (SELECT coalesce(max(step_index), -1) + 1 FROM generation_step)
WHERE (SELECT next_token_id FROM generated ORDER BY pos DESC LIMIT 1) <> 0;

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
