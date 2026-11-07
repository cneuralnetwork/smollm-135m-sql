SELECT 'shell' AS component,
       'SmolLM / SQL — Local inference studio' AS title,
       'SMOLLM / SQL' AS navbar_title,
       'database' AS icon,
       '/theme.css' AS css,
       'Run SmolLM-135M locally with an inference engine written in SQL.' AS description,
       'Local inference · Built with SQLPage' AS footer;

SELECT 'text' AS component,
       'SmolLM-135M' AS title,
       'A language model running inside SQLite. Enter a prompt, choose how many tokens to generate, and watch the output arrive.' AS contents;

SELECT 'html' AS component,
       '<div class="workspace-grid"><section class="workspace-input">' AS html;

SELECT 'form' AS component,
       'run.sql' AS action,
       '01 / PROMPT' AS title,
       'Run model' AS validate,
       'prompt-panel' AS class;
SELECT 'prompt' AS name,
       'Prompt' AS label,
       'textarea' AS type,
       4 AS rows,
       'Hello' AS value,
       true AS required,
       'A short prompt is best: each prompt token takes a full model pass.' AS description;
SELECT 'tokens' AS name,
       'Output tokens' AS label,
       'number' AS type,
       2 AS value,
       1 AS min,
       8 AS max,
       1 AS step,
       'Choose 1–8. Each token takes about 45 seconds on the tested machine.' AS description;

SELECT 'html' AS component,
       '</section><section class="workspace-output">' AS html;

SELECT 'text' AS component, '02 / OUTPUT' AS title;
SELECT CASE WHEN EXISTS (SELECT 1 FROM generated)
            THEN 'Prompt: ' || (SELECT text FROM prompt_input ORDER BY prompt_id DESC LIMIT 1)
            ELSE 'No output yet. Enter a prompt above to start.'
       END AS contents;

SELECT 'code' AS component, 'result-console' AS class
WHERE EXISTS (SELECT 1 FROM generated);
SELECT 'Generated text' AS title, text AS contents
FROM generated_text WHERE EXISTS (SELECT 1 FROM generated);

SELECT 'button' AS component, 'session-actions' AS class;
SELECT 'Generate one more token' AS title,
       'next.sql' AS link,
       'plus' AS icon,
       'primary' AS color
WHERE EXISTS (SELECT 1 FROM generated WHERE next_token_id <> 0);
SELECT 'Reset session' AS title,
       'reset.sql' AS link,
       'refresh' AS icon
WHERE EXISTS (SELECT 1 FROM generated);

SELECT 'text' AS component;
SELECT 'Tokens can be whole words or word fragments. The output panel shows the decoded text.' AS contents;

SELECT 'html' AS component, '</section></div>' AS html;

SELECT 'card' AS component, 3 AS columns, 'metric-row' AS class;
SELECT '135M' AS title, 'model parameters' AS description, 'database' AS icon;
SELECT '30' AS title, 'transformer layers' AS description;
SELECT 'SQL' AS title, 'tokenization through decoding' AS description, 'code' AS icon;
