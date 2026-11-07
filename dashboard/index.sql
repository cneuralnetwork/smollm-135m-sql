SELECT 'shell' AS component,
       'SmolLM · SQL Studio' AS title,
       'SmolLM · SQL Studio' AS navbar_title;

SELECT 'text' AS component,
       'SmolLM-135M SQL Studio' AS title,
       'Write a prompt and run the model in your local SQLite database.' AS contents;

SELECT 'form' AS component,
       'run.sql' AS action,
       'New prompt' AS title,
       'Run inference' AS validate;
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

SELECT 'text' AS component, 'Current session' AS title;
SELECT CASE WHEN EXISTS (SELECT 1 FROM generated)
            THEN 'Prompt: ' || (SELECT text FROM prompt_input ORDER BY prompt_id DESC LIMIT 1)
            ELSE 'No output yet. Enter a prompt above to start.'
       END AS contents;

SELECT 'code' AS component WHERE EXISTS (SELECT 1 FROM generated);
SELECT 'Generated text' AS title, text AS contents
FROM generated_text WHERE EXISTS (SELECT 1 FROM generated);

SELECT 'button' AS component;
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
SELECT 'Each click advances the SQL engine by one token. A token may be part of a word; the output above shows decoded text.' AS contents;

SELECT 'card' AS component, 3 AS columns;
SELECT '135M' AS title, 'model parameters' AS description, 'database' AS icon;
SELECT '30' AS title, 'transformer layers' AS description;
SELECT 'SQL' AS title, 'tokenization through decoding' AS description, 'code' AS icon;
