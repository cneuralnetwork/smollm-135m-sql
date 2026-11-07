-- Run exactly one more SQL inference step, then show the full decoded output.
SELECT 'shell' AS component,
       'Generating · SmolLM / SQL' AS title,
       'SMOLLM / SQL' AS navbar_title,
       'database' AS icon,
       '/theme.css' AS css,
       'Local inference · Built with SQLPage' AS footer;
SELECT 'text' AS component,
       'SmolLM-135M' AS title,
       'SQLite is running one more forward pass. The updated text will appear beside your prompt.' AS contents;

SELECT 'html' AS component,
       '<div class="workspace-grid"><section class="workspace-input">' AS html;
SELECT 'form' AS component,
       'run.sql' AS action,
       '01 / PROMPT' AS title,
       'Run again' AS validate,
       'prompt-panel' AS class;
SELECT 'prompt' AS name,
       'Prompt' AS label,
       'textarea' AS type,
       4 AS rows,
       (SELECT text FROM prompt_input ORDER BY prompt_id DESC LIMIT 1) AS value,
       true AS required,
       'Edit the prompt and run again after this token finishes.' AS description;
SELECT 'tokens' AS name,
       'Output tokens' AS label,
       'number' AS type,
       2 AS value,
       1 AS min,
       8 AS max,
       1 AS step,
       'Choose 1–8 output tokens.' AS description;
SELECT 'html' AS component,
       '</section><section class="workspace-output live-output">' AS html;
SELECT 'text' AS component, '02 / OUTPUT' AS title;

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
SELECT 'html' AS component, '</section></div>' AS html;
