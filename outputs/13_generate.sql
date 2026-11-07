-- After prompt IDs have been inserted, one generation_request advances the
-- model for any number of additional positions, entirely through SQL triggers.
-- For N output tokens, request N-1 additional steps; the last prompt position
-- has already produced the first output token.
PRAGMA recursive_triggers = ON;
CREATE TABLE IF NOT EXISTS generation_step (
  step_index INTEGER PRIMARY KEY CHECK (step_index >= 0)
);
CREATE TABLE IF NOT EXISTS generation_request (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  additional_steps INTEGER NOT NULL CHECK (additional_steps BETWEEN 0 AND 2047)
);

CREATE TRIGGER IF NOT EXISTS run_generation_step
AFTER INSERT ON generation_step
BEGIN
  SELECT RAISE(ABORT, 'insert prompt tokens before generating')
  WHERE NOT EXISTS (SELECT 1 FROM generated);
  INSERT INTO inference_input(pos, token_id)
  SELECT pos + 1, next_token_id
  FROM (SELECT pos, next_token_id FROM generated
        ORDER BY pos DESC LIMIT 1)
  WHERE next_token_id <> 0;
END;

CREATE TRIGGER IF NOT EXISTS run_generation_request
AFTER INSERT ON generation_request
BEGIN
  DELETE FROM generation_step;
  INSERT INTO generation_step(step_index)
  WITH RECURSIVE numbers(n) AS (
    SELECT 0 WHERE NEW.additional_steps > 0
    UNION ALL
    SELECT n + 1 FROM numbers WHERE n + 1 < NEW.additional_steps
  )
  SELECT n FROM numbers;
END;
